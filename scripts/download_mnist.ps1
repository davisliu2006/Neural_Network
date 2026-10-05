# Download MNIST IDX files from Azure Open Datasets into data/mnist/
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$Root = Split-Path -Parent $ScriptDir
$OutDir = Join-Path $Root "data\mnist"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Base = "https://azureopendatastorage.blob.core.windows.net/mnist"
$Files = @(
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
)

function Expand-GzipFile([string]$GzPath, [string]$DestPath) {
    $inStream = [System.IO.File]::OpenRead($GzPath)
    try {
        $gzip = New-Object System.IO.Compression.GzipStream(
            $inStream, [System.IO.Compression.CompressionMode]::Decompress)
        try {
            $outStream = [System.IO.File]::Create($DestPath)
            try { $gzip.CopyTo($outStream) }
            finally { $outStream.Dispose() }
        } finally { $gzip.Dispose() }
    } finally { $inStream.Dispose() }
}

foreach ($name in $Files) {
    $url = "$Base/$name"
    $gzPath = Join-Path $OutDir $name
    $outName = $name -replace '\.gz$', ''
    $outPath = Join-Path $OutDir $outName

    if (Test-Path $outPath) {
        Write-Host "exists: $outName"
        continue
    }

    Write-Host "download: $name"
    Invoke-WebRequest -Uri $url -OutFile $gzPath

    Write-Host "inflate:  $outName"
    Expand-GzipFile $gzPath $outPath
    Remove-Item $gzPath
}

Write-Host "done: $OutDir"
