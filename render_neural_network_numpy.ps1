$ErrorActionPreference = "Stop"

$taskRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskTexBin = Join-Path $taskRoot ".tinytex\TinyTeX\bin\windows"
$taskLocalManim = Join-Path $taskRoot ".venv\Scripts\manim.exe"
$taskScene = Join-Path $taskRoot "neural_network_learning_numpy.py"

if (Test-Path -LiteralPath (Join-Path $taskTexBin "latex.exe")) {
    $env:PATH = "$taskTexBin;$env:PATH"
}

if (Test-Path -LiteralPath $taskLocalManim) {
    $taskManim = $taskLocalManim
}
else {
    $taskManimCommand = Get-Command manim -ErrorAction SilentlyContinue
    if ($null -eq $taskManimCommand) {
        throw "Manim não encontrado. Instale as dependências ou crie o ambiente .venv."
    }
    $taskManim = $taskManimCommand.Source
}

if (-not (Test-Path -LiteralPath $taskScene)) {
    throw "Cena NumPy não encontrada em $taskScene."
}

if ($null -eq (Get-Command latex -ErrorAction SilentlyContinue)) {
    throw "LaTeX não encontrado no PATH. Instale uma distribuição TeX compatível com o Manim."
}

if ($null -eq (Get-Command dvisvgm -ErrorAction SilentlyContinue)) {
    throw "dvisvgm não encontrado no PATH. Ele é necessário para renderizar MathTex."
}

Push-Location $taskRoot
try {
    & $taskManim `
        -r 1080,1350 `
        --fps 60 `
        $taskScene `
        NeuralNetworkLearningNumpy `
        -o ComoUmaRedeNeuralAprende_NumPy_corrigido_1080x1350.mp4

    if ($LASTEXITCODE -ne 0) {
        throw "O Manim encerrou com o código $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
