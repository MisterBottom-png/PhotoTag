param(
    [string]$Version = "1.22.0"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root "vendor\onnxruntime\win-x64-directml"
$tempDir = Join-Path $env:TEMP "onnxruntime-directml-$Version"
$nupkg = Join-Path $tempDir "Microsoft.ML.OnnxRuntime.DirectML.$Version.nupkg"
$zip = Join-Path $tempDir "Microsoft.ML.OnnxRuntime.DirectML.$Version.zip"
$url = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$Version"

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $nupkg

Write-Host "Extracting $nupkg"
Copy-Item $nupkg $zip -Force
Expand-Archive -Path $zip -DestinationPath $tempDir -Force

$runtimeDir = Join-Path $tempDir "runtimes\win-x64\native"
if (-not (Test-Path $runtimeDir)) {
    throw "Expected runtime dir not found: $runtimeDir"
}

Get-ChildItem -Path $runtimeDir -Filter *.dll | ForEach-Object {
    Copy-Item $_.FullName -Destination $vendorDir -Force
}

Write-Host "DirectML DLLs copied to $vendorDir"
