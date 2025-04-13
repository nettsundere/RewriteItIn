open System
open System.Buffers
open System.Buffers.Binary
open System.IO
open System.Net.Http
open System.Numerics
open System.Text
open System.Text.Json
open System.Text.Json.Serialization

let modelGlobal = "deepseek-chat"
let batchSizeGlobal = 30
let maxTokensGlobal = 8_192
let globalTimeoutMinutes = 5;
let globalTemperature = 1.8

type FileData = {
    filename: string
    source: string
}

type JsonSchema = {
    ``type``: string
    properties: Map<string, JsonSchema> option
    required: string list option
}

type JsonSchemaWrapper = {
    name: string
    strict: string
    schema: JsonSchema
}

type ResponseFormatType =
    | JsonObject
    | JsonSchema of JsonSchemaWrapper

type ResponseFormat = {
    [<JsonPropertyName("type")>]
    ``type``: string
    [<JsonPropertyName("json_schema")>]
    json_schema: JsonSchemaWrapper option
}

type ChatMessage = {
    role: string
    content: string
}

type ChatRequest = {
    model: string
    messages: ChatMessage list
    temperature: float
    max_tokens: int
    stream: bool
    response_format: ResponseFormat option
}

type CodeFile = {
    path: string
    content: string
}

type AnalysisContext = {
    files: FileData list
    sourceLang: string
    targetLang: string
    principles: string option
}

type Options = {
    SourcePath: string
    SourceLang: string
    TargetPath: string
    TargetLang: string
    ServerUrl: string
    Principles: string option
    ApiKey: string option
    UseJsonSchema: bool option
}

let createResponseFormat (formatType: ResponseFormatType) =
    match formatType with
    | JsonObject ->
        {
            ``type`` = "json_object"
            json_schema = None
        }
    | JsonSchema schema ->
        {
            ``type`` = "json_schema"
            json_schema = Some schema
        }

let createFileObjectSchema() =
    {
        ``type`` = "object"
        properties = Some (Map [
            ("filename", { 
                ``type`` = "string"
                properties = None
                required = None 
            })
            ("source", { 
                ``type`` = "string"
                properties = None
                required = None 
            })
        ])
        required = Some ["filename"; "source"]
    }

let createFilesResponseSchema() =
    {
        name = "files_response"
        strict = "true"
        schema = {
            ``type`` = "object"
            properties = Some (Map [
                ("files", {
                    ``type`` = "array"
                    properties = None
                    required = None
                })
            ])
            required = Some ["files"]
        }
    }

let isBinaryFile (filePath: string) =
    try
        use fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read)
        use memory = MemoryPool<byte>.Shared.Rent(1024)
        let span = memory.Memory.Span
        let bytesRead = fs.Read(span)
        
        if bytesRead = 0 then false
        else
            let isBinary = 
                // Check for text file markers first
                match span with
                | _ when bytesRead >= 3 && span[0] = 0xEFuy && span[1] = 0xBBuy && span[2] = 0xBFuy -> false // UTF-8 BOM
                | _ when bytesRead >= 2 ->
                    match span[0], span[1] with
                    | 0xFEuy, 0xFFuy | 0xFFuy, 0xFEuy -> false // UTF-16 BOMs
                    | _ ->
                        // Check for binary markers
                        let mutable i = 0
                        let mutable foundBinary = false
                        
                        while not foundBinary && i < bytesRead do
                            let c = span[i]
                            if c = 0uy then
                                foundBinary <- true
                            elif c > 0x7Fuy then
                                let leadingOnes = BitOperations.LeadingZeroCount(uint32 (~~~(int c)) <<< 24)
                                if leadingOnes < 2 || leadingOnes > 4 then
                                    foundBinary <- true
                                elif i + leadingOnes > bytesRead then
                                    foundBinary <- true
                                else
                                    let mutable j = 1
                                    while not foundBinary && j < leadingOnes do
                                        if span[i + j] &&& 0xC0uy <> 0x80uy then
                                            foundBinary <- true
                                        j <- j + 1
                                    i <- i + leadingOnes - 1
                            i <- i + 1
                        
                        foundBinary
                | _ -> false

            isBinary || 
            // Check for binary file signatures if no markers found
            (bytesRead >= 4 && 
                let signature = BinaryPrimitives.ReadUInt32LittleEndian(span)
                match signature with
                | 0x46445025u  // %PDF
                | 0x04034B50u   // ZIP
                | 0x464C457Fu   // ELF
                | 0x4D5A9000u   // MZ (PE)
                | 0x474E5089u   // PNG
                | 0x38464947u   // GIF
                | 0x46464952u   // RIFF
                | 0xE011CFD0u -> true  // MS Office
                | _ -> false)
    with
    | _ -> true
let readSourceFiles (sourcePath: string) (allowedFormats: string list option) =
    let getRelativePath (filePath: string) =
        if Directory.Exists sourcePath then
            Path.GetRelativePath(sourcePath, filePath)
        else
            Path.GetFileName(filePath)

    let filterByFormat (files: string array) =
        match allowedFormats with
        | Some formats ->
            files |> Array.filter (fun f ->
                formats |> List.exists (fun ext ->
                    f.EndsWith(ext, StringComparison.OrdinalIgnoreCase)
                ))
        | _ -> files

    if Directory.Exists sourcePath then
        Directory.GetFiles(sourcePath, "*", SearchOption.AllDirectories)
        |> Array.filter (isBinaryFile >> not)
        |> filterByFormat
        |> Array.map (fun filePath -> 
            let relativePath = getRelativePath filePath
            let content = File.ReadAllText(filePath)
            { path = relativePath; content = content })
    elif File.Exists sourcePath then
        if isBinaryFile sourcePath then
            [||]
        else
            let shouldInclude = 
                match allowedFormats with
                | Some formats -> formats |> List.exists (fun ext -> sourcePath.EndsWith(ext, StringComparison.OrdinalIgnoreCase))
                | None -> true
            if shouldInclude then
                [| { path = getRelativePath sourcePath; content = File.ReadAllText(sourcePath) } |]
            else
                [||]
    else
        failwith "Source path does not exist"

let createFormatQueryPayload (sourceLang: string) (targetLang: string) (useJsonSchema: bool) =
    let systemMessage = {
        role = "system"
        content = "Keep the dot, strip whitespace. Response must be JSON with this exact structure: {\"formats\": [\".ext1\", \".ext2\"]}"
    }

    let userMessage = {
        role = "user"
        content = $"List all file extensions are used when writing apps using {sourceLang} OR {targetLang}, ignore binary files"
    }

    let responseFormat = 
        if useJsonSchema then
            Some (createResponseFormat (JsonSchema {
                name = "formats_response"
                strict = "true"
                schema = {
                    ``type`` = "object"
                    properties = Some (Map [
                        ("formats", {
                            ``type`` = "array"
                            properties = None
                            required = None
                        })
                    ])
                    required = Some ["formats"]
                }
            }))
        else
            Some (createResponseFormat JsonObject)

    {
        model = modelGlobal
        messages = [systemMessage; userMessage]
        temperature = globalTemperature
        max_tokens = maxTokensGlobal
        stream = false
        response_format = responseFormat
    }

let createAnalysisPayload (files: CodeFile list) (useJsonSchema: bool) =
    let systemMessage = {
        role = "system"
        content = "Analyze these code files and identify which ones are important for the codebase. Respond with a valid JSON object containing a 'files' array with objects acknowledging only the important files that need to be rewritten. Each object should have 'filename' and 'source' fields. Source should be set as \"ok\" for important files."
    }

    let fileDataList = 
        files 
        |> List.map (fun f -> { filename = f.path; source = f.content })

    let userContent = 
        JsonSerializer.Serialize({| files = fileDataList |}, JsonSerializerOptions(WriteIndented = true))

    let userMessage = {
        role = "user"
        content = userContent
    }

    let responseFormat = 
        if useJsonSchema then
            Some (createResponseFormat (JsonSchema (createFilesResponseSchema())))
        else
            Some (createResponseFormat JsonObject)

    {
        model = modelGlobal
        messages = [systemMessage; userMessage]
        temperature = globalTemperature
        max_tokens = maxTokensGlobal
        stream = false
        response_format = responseFormat
    }

let createRewritePayload (files: CodeFile list) (context: AnalysisContext) (useJsonSchema: bool) =
    let principlesPart = 
        context.principles 
        |> Option.map (fun p -> $"Follow these mandatory principles: {p}.")
        |> Option.defaultValue ""
        
    let systemMessage = {
        role = "system"
        content =
            $"You have to rewrite code in {context.targetLang} using the source in {context.sourceLang} as example." + principlesPart + "User input is JSON. Always respond with complete, unabridged answers. What you respond with is the final result. Do not simplify for brevity. You must respond with a valid JSON object containing a 'files' array with objects having 'filename' and 'source' fields. Rewrite the code as requested. It is ok to ignore some files and change filenames and create many files instead of one or many source files when it makes sense. The response MUST be valid JSON with this exact structure: {\"files\": [{\"filename\": \"target/path.ext\", \"source\": \"rewritten code\"}, ...]}"
    }

    let fileDataList = files |> List.map (fun f -> { filename = f.path; source = f.content })
    
    let userContent = JsonSerializer.Serialize({| files = fileDataList |})

    let userMessage = {
        role = "user"
        content = userContent
    }

    let responseFormat = 
        if useJsonSchema then
            Some (createResponseFormat (JsonSchema (createFilesResponseSchema())))
        else
            Some (createResponseFormat JsonObject)

    {
        model = modelGlobal
        messages = [systemMessage; userMessage]
        temperature = globalTemperature
        max_tokens = maxTokensGlobal
        stream = false
        response_format = responseFormat
    }

let sendRequest (serverUrl: string) (apiKey: string option) (payload: ChatRequest) =
    async {
        use client = new HttpClient()
        client.Timeout <- TimeSpan.FromMinutes(globalTimeoutMinutes)
        let url = $"{serverUrl}/v1/chat/completions"
        
        let options = JsonSerializerOptions()
        options.DefaultIgnoreCondition <- JsonIgnoreCondition.WhenWritingNull
        let json = JsonSerializer.Serialize(payload, options)
        
        let content = new StringContent(json, Encoding.UTF8, "application/json")
        
        match apiKey with
        | Some key -> client.DefaultRequestHeaders.Add("Authorization", $"Bearer {key}")
        | None -> ()
        
        printfn $"Sending request to: {url}"

        let! response = client.PostAsync(url, content) |> Async.AwaitTask
        
        if not response.IsSuccessStatusCode then
            let! errorContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
            failwith $"Request failed with status {response.StatusCode}: {errorContent}"

        let! responseContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
        printfn $"Received response: {responseContent}"
        return responseContent
    }

let parseFormatsResponse (responseContent: string) =
    try
        use jsonDoc = JsonDocument.Parse(responseContent)
        let root = jsonDoc.RootElement
        
        match root.TryGetProperty("choices") with
        | (true, choices) when choices.ValueKind = JsonValueKind.Array && choices.GetArrayLength() > 0 ->
            let firstChoice = choices[0]
            
            match firstChoice.TryGetProperty("message") with
            | (true, message) ->
                match message.TryGetProperty("content") with
                | (true, content) when content.ValueKind = JsonValueKind.String ->
                    let contentStr = content.GetString()
                    try
                        let parsed = JsonDocument.Parse(contentStr).RootElement
                        match parsed.TryGetProperty("formats") with
                        | (true, formats) when formats.ValueKind = JsonValueKind.Array ->
                            let results = JsonSerializer.Deserialize<string list>(formats.GetRawText())
                            if List.isEmpty results then
                                printfn "No formats specified, will process all files"
                                None
                            else
                                Some results
                        | _ ->
                            printfn "No formats array found in response"
                            None
                    with ex ->
                        printfn $"Failed to deserialize formats response: {ex.Message}"
                        None
                | _ -> 
                    printfn "Invalid response: missing content"
                    None
            | _ -> 
                printfn "Invalid response: missing message"
                None
        | _ -> 
            printfn "Invalid response: missing choices"
            None
    with ex ->
        printfn $"Error parsing formats response: {ex.Message}"
        None

let parseResponse (responseContent: string) =
    try
        use jsonDoc = JsonDocument.Parse(responseContent)
        let root = jsonDoc.RootElement
        
        match root.TryGetProperty("choices") with
        | (true, choices) when choices.ValueKind = JsonValueKind.Array && choices.GetArrayLength() > 0 ->
            let firstChoice = choices[0]
            
            match firstChoice.TryGetProperty("message") with
            | (true, message) ->
                match message.TryGetProperty("content") with
                | (true, content) when content.ValueKind = JsonValueKind.String ->
                    let contentStr = content.GetString()
                    try
                        let parsed = JsonDocument.Parse(contentStr).RootElement
                        match parsed.TryGetProperty("files") with
                        | (true, files) when files.ValueKind = JsonValueKind.Array ->
                            let results = JsonSerializer.Deserialize<FileData list>(files.GetRawText())
                            Some results
                        | _ ->
                            printfn "Invalid response: missing files array"
                            None
                    with ex ->
                        printfn $"Failed to deserialize response: {ex.Message}"
                        None
                | _ -> 
                    printfn "Invalid response: missing content"
                    None
            | _ -> 
                printfn "Invalid response: missing message"
                None
        | _ -> 
            printfn "Invalid response: missing choices"
            None
    with ex ->
        printfn $"Error parsing response: {ex.Message}"
        None

let writeOutputFile (targetPath: string) (file: FileData) =
    let fullPath = Path.Combine(targetPath, file.filename)
    let directory = Path.GetDirectoryName(fullPath)
    
    if not (String.IsNullOrEmpty(directory)) then
        Directory.CreateDirectory(directory) |> ignore
    
    File.WriteAllText(fullPath, file.source)
    printfn $"Written: {file.filename}"

let parseCommandLine (args: string[]) =
    let rec parseInternal (args: string list) (options: Options) =
        match args with
        | [] -> options
        | "--source" :: src :: rest -> parseInternal rest { options with SourcePath = src }
        | "--source-lang" :: lang :: rest -> parseInternal rest { options with SourceLang = lang }
        | "--target" :: tgt :: rest -> parseInternal rest { options with TargetPath = tgt }
        | "--target-lang" :: lang :: rest -> parseInternal rest { options with TargetLang = lang }
        | "--server" :: url :: rest -> parseInternal rest { options with ServerUrl = url }
        | "--principles" :: principles :: rest -> parseInternal rest { options with Principles = Some principles }
        | "--api-key" :: key :: rest -> parseInternal rest { options with ApiKey = Some key }
        | "--use-json-schema" :: rest -> parseInternal rest { options with UseJsonSchema = Some true }
        | x :: _ -> failwith $"Unknown option: {x}"
    
    let defaultOptions = {
        SourcePath = ""
        SourceLang = ""
        TargetPath = ""
        TargetLang = ""
        ServerUrl = ""
        Principles = None
        ApiKey = None
        UseJsonSchema = None
    }
    
    parseInternal (args |> Array.toList) defaultOptions

let queryRelevantFormats serverUrl apiKey sourceLang targetLang useJsonSchema =
    printfn "\n=== Querying relevant file formats ==="
    let payload = createFormatQueryPayload sourceLang targetLang useJsonSchema
    let response = sendRequest serverUrl apiKey payload |> Async.RunSynchronously
    parseFormatsResponse response

let printBatch list =
    printfn $"=== Batch ==="
    list 
    |> List.iter (fun file -> printfn $"  - {file.path}")

let filterImportantFiles serverUrl apiKey (allFiles: CodeFile[]) useJsonSchema =
    printfn "\n=== Filtering important files ==="
    let batchSize = batchSizeGlobal
    let batches = 
        allFiles 
        |> Array.toList
        |> List.chunkBySize batchSize

    let importantFiles = ResizeArray<CodeFile>()

    for batch in batches do
        printfn $"Analyzing batch of {batch.Length} files..."
        printBatch batch
        let payload = createAnalysisPayload batch useJsonSchema
        let response = sendRequest serverUrl apiKey payload |> Async.RunSynchronously
        
        match parseResponse response with
        | Some results -> 
            let importantInBatch = 
                results 
                |> List.filter (fun f -> f.source = "ok")
                |> List.map (fun f -> 
                    batch |> List.find (fun src -> src.path = f.filename))
            
            importantFiles.AddRange(importantInBatch)
            printfn $"Found {importantInBatch.Length} important files in this batch"
        | None -> 
            printfn "No important files identified in this batch"

    importantFiles |> Seq.toArray

let rewriteFiles serverUrl apiKey (context: AnalysisContext) (files: CodeFile list) useJsonSchema =
    printfn $"\nProcessing {files.Length} files..."
    try
        let payload = createRewritePayload files context useJsonSchema
        let response = sendRequest serverUrl apiKey payload |> Async.RunSynchronously
        
        match parseResponse response with
        | Some results ->
            printfn $"Successfully converted {results.Length} files"
            Some results
        | None ->
            printfn "No valid conversion returned for these files"
            None
    with ex ->
        printfn $"Error processing files: {ex.Message}"
        None

let runConversion (options: Options) =
    try
        let useJsonSchema = options.UseJsonSchema |> Option.defaultValue false
        printfn $"Determining relevant file formats for {options.SourceLang} -> {options.TargetLang} conversion"
        let relevantFormatsList = queryRelevantFormats options.ServerUrl options.ApiKey options.SourceLang options.TargetLang useJsonSchema
        let relevantFormats = match relevantFormatsList with Some f -> String.Join(", ", f) | None -> "All files"
        printfn $"Relevant formats: {relevantFormats}"

        printfn $"Reading source files from %s{options.SourcePath}"
        let allSourceFiles = readSourceFiles options.SourcePath relevantFormatsList
        printfn $"Found {allSourceFiles.Length} relevant files to process"

        // Filter to get only important files
        let importantFiles = filterImportantFiles options.ServerUrl options.ApiKey allSourceFiles useJsonSchema
        printfn $"Identified {importantFiles.Length} important files to convert"

        let context = {
            files = importantFiles |> Array.map (fun f -> { filename = f.path; source = f.content }) |> Array.toList
            sourceLang = options.SourceLang
            targetLang = options.TargetLang
            principles = options.Principles
        }
        
        Directory.CreateDirectory(options.TargetPath) |> ignore
        let mutable successCount = 0

        printfn "\n=== Starting rewrite phase ==="
        let batchSize = batchSizeGlobal
        let batches = 
            importantFiles 
            |> Array.toList
            |> List.chunkBySize batchSize

        for batch in batches do
            printBatch batch
            match rewriteFiles options.ServerUrl options.ApiKey context batch useJsonSchema with
            | Some results ->
                results |> List.iter (fun result ->
                    writeOutputFile options.TargetPath result
                    successCount <- successCount + 1
                )
            | None -> ()

        printfn $"\nConversion complete. Successfully converted {successCount} of {importantFiles.Length} important files"
        0
    with
    | :? OperationCanceledException as ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn $"\nTask cancelled: {ex.Message}"
        Console.ResetColor()
        1
    | ex ->
        printfn $"\nError: %s{ex.Message}"
        1

[<EntryPoint>]
let main argv =
    try
        if argv.Length = 0 then
            printfn "Usage: dotnet run [options]"
            printfn "Options:"
            printfn "  --source <path>        Source file or directory"
            printfn "  --source-lang <lang>   Source language"
            printfn "  --target <path>        Target directory"
            printfn "  --target-lang <lang>   Target language"
            printfn "  --server <url>         API server URL"
            printfn "  --principles <text>    (Optional) Conversion principles"
            printfn "  --api-key <key>        (Optional) API key for authentication"
            printfn "  --use-json-schema      (Optional) Use JSON Schema"
            1
        else
            let options = parseCommandLine argv
            
            if String.IsNullOrEmpty(options.SourcePath) then failwith "Source path is required"
            if String.IsNullOrEmpty(options.SourceLang) then failwith "Source language is required"
            if String.IsNullOrEmpty(options.TargetPath) then failwith "Target path is required"
            if String.IsNullOrEmpty(options.TargetLang) then failwith "Target language is required"
            if String.IsNullOrEmpty(options.ServerUrl) then failwith "Server URL is required"
            
            runConversion options
    with
    | ex ->
        printfn $"Error: {ex.Message}"
        1