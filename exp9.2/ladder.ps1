# exp9.2 overnight ladder: waits for a running agent.py (the pilot) to finish, then
# runs every model on all 10 tasks, both groups. Episodes already in out/episodes are skipped.
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-File','ladder.ps1'
Set-Location $PSScriptRoot
$models = 'tev1:4b', 'gemma4:e2b', 'gemma3:1b', 'qwen3:4b-instruct-2507-q4_K_M', 'gemma3:4b-it-q4_K_M',
          'ornith:9b', 'qwen3.6:35b-a3b'
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*agent.py*' }) {
    Start-Sleep -Seconds 30
}
"$(Get-Date -Format s) ladder start" | Out-File -Append -Encoding utf8 out_ladder.log
foreach ($m in $models) {
    python -u agent.py --models $m 2>&1 | Out-File -Append -Encoding utf8 out_ladder.log
    "$(Get-Date -Format s) $m finished" | Out-File -Append -Encoding utf8 out_ladder.log
}
"$(Get-Date -Format s) ladder done" | Out-File -Append -Encoding utf8 out_ladder.log
