# exp9.4 driver: models x tasks, one headless 1bcoder run per pair from the task's checkout.
# The model calls go to -OllamaHost; indexes and 1bcoder run here. Done pairs (review + stats) are skipped.
#   powershell -File run.ps1 -OllamaHost http://localhost:11434 -Models qwen3:1.7b
#   powershell -File run.ps1 -OllamaHost http://panteon.local:11434 -Models gemma4:e4b,gpt-oss:20b
param([string]$OllamaHost = 'http://localhost:11434', [string[]]$Models = @(), [int]$Ctx = 8192,
      [string[]]$Tasks = @('T01','T02','T03','T04','T05','T06','T07','T08','T09','T10'))
Set-Location $PSScriptRoot
# redirected output from cmd would otherwise be cp1251; 1bcoder's box characters need UTF-8
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$runModels = @($Models | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$runTasks = @($Tasks | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$hostTag = ([uri]$OllamaHost).Host
New-Item -ItemType Directory -Force out | Out-Null
# 1bcoder writes its #POS resume line into the script file: each driver gets its own copy
$script = "$PSScriptRoot\out\.pipeline-$hostTag.txt"
Copy-Item "$PSScriptRoot\pipeline.txt" $script -Force
$script = $script -replace '\\', '/'
if (-not (Test-Path out\timing.csv)) { "model,task,seconds,exit,ctx,host" | Out-File -Encoding utf8 out\timing.csv }
function Say($text) { Write-Host $text; $text | Out-File -Append -Encoding utf8 out\run.log }

foreach ($m in $runModels) {
    $tag = $m -replace '[:/]', '_'
    # temperature 0 without a repeat penalty loops: gpt-oss in exp9.3 ("Ok." x769),
    # qwen3:1.7b here (207 ACTION lines in one reply) - one setting for every model
    $rp = '1.2'
    foreach ($t in $runTasks) {
        if ((Test-Path "out\$tag\review-$t.txt") -and (Test-Path "out\$tag\stats-$t.json")) { continue }
        New-Item -ItemType Directory -Force "out\$tag" | Out-Null
        Say "$(Get-Date -Format s) $m $t start ($hostTag)"
        Push-Location "C:\Projects\Data\wt\sonar\$t"
        $sw = [Diagnostics.Stopwatch]::StartNew()
        1bcoder --scriptapply $script --param id=$t --param tag=$tag --param ctx=$Ctx --param rp=$rp --model $m --host $OllamaHost *> "$PSScriptRoot\out\$tag\log-$t.txt"
        $code = $LASTEXITCODE
        $sw.Stop()
        Pop-Location
        $sec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        "$m,$t,$sec,$code,$Ctx,$hostTag" | Out-File -Append -Encoding utf8 out\timing.csv
        $note = if ($code -ne 0) { "  FAILED exit $code - see out\$tag\log-$t.txt" } else { "" }
        Say "$(Get-Date -Format s) $m $t ${sec}s$note"
    }
}
Say "$(Get-Date -Format s) done ($hostTag)"
