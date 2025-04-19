module RewriteItIn.Concurrency

open System

    let runThrottled (maxConcurrency: int) (tasks: seq<Async<'T>>) () = 
        async {
            use semaphore = new Threading.SemaphoreSlim(maxConcurrency)
            let results = ResizeArray<'T>()
            let exceptions = ResizeArray<exn>()

            let processTask task = async {
                try 
                    do! semaphore.WaitAsync() |> Async.AwaitTask
                    try
                        let! result = task
                        lock results (fun () -> results.Add(result))
                    finally
                        semaphore.Release() |> ignore
                with ex ->
                    lock exceptions (fun () -> exceptions.Add(ex))
            }

            let! completed = 
                tasks
                |> Seq.map processTask
                |> Async.Parallel

            if exceptions.Count > 0 then
                return raise (AggregateException(exceptions))
            
            return results
        }