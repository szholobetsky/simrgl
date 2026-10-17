# exp9.3 local run: every model on the same tasks, one 1bcoder headless run per (model, task),
# started from the task's checkout. Wall time goes to out/timing.csv; 1bcoder writes the answer
# (output-<id>.txt) and its own token statistics (stats-<id>.json). A pair with both files is
# skipped, so a stopped run resumes where it was.
#   start.bat     (or: powershell -File run_local.ps1 [-Ctx 8192])
#   start_panteon.bat   (the larger models on panteon: -OllamaHost http://panteon.local:11434 -Models a,b)
# The indexes (idx.py) and 1bcoder always run here; only the model calls go to the host.
param([int]$Ctx = 8192, [string]$OllamaHost = 'http://localhost:11434', [string[]]$Models = @())
Set-Location $PSScriptRoot
# redirected output from cmd/start.bat would otherwise be cp1251 and 1bcoder's box characters crash it
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
# PowerShell names are case-insensitive: the default list must not be called $models (= the -Models parameter)
$localModels = 'tev1:0.8b', 'gemma3:1b', 'qwen3:1.7b', 'qwen2.5-coder:3b-instruct', 'sam860/jamba-reasoning:3b',
               'nemotron-3-nano:4b', 'qwen2.5-coder:7b-instruct', 'lfm2.5:8b'
$runModels = @($Models | ForEach-Object { $_ -split ',' } | Where-Object { $_ })   # -File passes "a,b" as one string
if (-not $runModels.Count) { $runModels = $localModels }
$hostTag = ([uri]$OllamaHost).Host
$tasks  = 'T09', 'T05'
$script = "$PSScriptRoot\passive.txt" -replace '\\', '/'
New-Item -ItemType Directory -Force out | Out-Null
if (-not (Test-Path out\timing.csv)) { "model,task,seconds,exit,ctx,host" | Out-File -Encoding utf8 out\timing.csv }

function Say($text) {
    # Tee-Object in PowerShell 5.1 writes UTF-16; Out-File -Encoding utf8 keeps the log readable
    Write-Host $text
    $text | Out-File -Append -Encoding utf8 out\run_local.log
}

foreach ($m in $runModels) {
    $tag = $m -replace '[:/]', '_'
    foreach ($t in $tasks) {
        if ((Test-Path "out\$tag\output-$t.txt") -and (Test-Path "out\$tag\stats-$t.json")) { continue }
        New-Item -ItemType Directory -Force "out\$tag" | Out-Null
        $log = "$PSScriptRoot\out\$tag\log-$t.txt"
        Say "$(Get-Date -Format s) $m $t start (ctx $Ctx, $hostTag)"
        Push-Location "C:\Projects\Data\wt\sonar\$t"
        $sw = [Diagnostics.Stopwatch]::StartNew()
        1bcoder --scriptapply $script --param id=$t --param tag=$tag --param ctx=$Ctx --model $m --host $OllamaHost *> $log
        $code = $LASTEXITCODE
        $sw.Stop()
        Pop-Location
        $sec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        "$m,$t,$sec,$code,$Ctx,$hostTag" | Out-File -Append -Encoding utf8 out\timing.csv
        $note = if ($code -ne 0) { "  FAILED exit $code - see $log" } else { "" }
        Say "$(Get-Date -Format s) $m $t ${sec}s$note"
    }
}
Say "$(Get-Date -Format s) done"
