module RewriteItIn.Concurrency

open System
open System.Collections.Concurrent
open System.Threading

let runThrottled (maxConcurrency: int) (tasks: seq<Async<'T>>) : Async<'T list> = 
    async {
        use semaphore = new SemaphoreSlim(maxConcurrency)
        let results = ConcurrentBag<'T>()
        let exceptions = ConcurrentBag<exn>()

        let processTask (task: Async<'T>) = async {
            try
                do! semaphore.WaitAsync() |> Async.AwaitTask
                try
                    let! result = task
                    results.Add(result)
                finally
                    semaphore.Release() |> ignore
            with ex ->
                exceptions.Add(ex)
        }

        let! _ =
            tasks
            |> Seq.map processTask
            |> Async.Parallel

        if exceptions.Count > 0 then
            return raise (AggregateException(exceptions))
        
        return results.ToArray() |> Array.toList
    }