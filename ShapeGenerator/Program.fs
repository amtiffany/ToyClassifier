open System
open SixLabors.ImageSharp
open SixLabors.ImageSharp.PixelFormats
open SixLabors.ImageSharp.Processing
open SixLabors.ImageSharp.Drawing.Processing
open SixLabors.ImageSharp.Drawing

let width = 400
let height = 400 



let imageShape (radius:float32) (center:PointF) (useRect:bool) (useStar:bool)=
    // Create a new image with white background
    let image = new Image<Rgba32>(width, height)
    image.Mutate(fun ctx ->
        ctx.Fill(Color.White) |> ignore
        // Draw a filled black circle
        let rect = new RectangleF(center,Size(int radius,int radius))
        let star = new Star(center, 4, float32 radius * (float32 0.5), radius)
        let circle = EllipsePolygon(center, radius)
        if useRect then
            ctx.Fill(Color.Black, rect) |> ignore
        else if useStar then
            ctx.Fill(Color.Black, star) |> ignore
        else
            ctx.Fill(Color.Black, circle) |> ignore

    )
    
    image





[<EntryPoint>]
let main argv =

//    printfn "%A" argv

    let useRect = argv.Length > 0 && argv[0] = "r"




    let rnd = Random()
    let N = 10

    for i in 1..N do 
        let h = rnd.NextDouble() |> float32
        let w = rnd.NextDouble() |> float32

        let center = PointF(w * (float32 width), h * (float32 height) / 2.0f)

        let mx = float <| max width height 

        let radius = (rnd.NextDouble() * mx * 0.8) |> float32

        printfn "%A, %A" center radius

        // Save to PNG
        let outputPath = $"img/img{i}.png"
        use image = imageShape radius center useRect (not useRect)
        image.Save(outputPath)
        printfn $"Image saved to {outputPath}"
    0