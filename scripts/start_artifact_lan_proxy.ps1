$ErrorActionPreference = 'Stop'
# Run at Windows login. WSL may receive a new IP after a restart.
$python = 'C:\Users\chenws\AppData\Local\Programs\Python\Python310\python.exe'
$repo = '\\wsl.localhost\Ubuntu\home\chenws\code\RGB_SFM_V3'
$addressText = (& wsl.exe -d Ubuntu -- hostname -I) -join ' '
$wslAddress = ($addressText.Trim() -split '\s+')[0]
if (-not $wslAddress) { throw 'Cannot determine WSL address' }
if (Get-NetTCPConnection -State Listen -LocalPort 40115 -ErrorAction SilentlyContinue) {
    Write-Output 'Port 40115 already listening; leaving the existing process intact.'
    exit 0
}
$arguments = @('-u', "$repo\scripts\artifact_lan_proxy.py", '--target', $wslAddress,
               '--allow', '140.115.51.165/32', '--allow', '192.168.100.0/24',
               '--allow', '172.16.0.0/12')
$process = Start-Process -FilePath $python -ArgumentList $arguments -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput 'D:\RGB_SFM_artifacts\lan_proxy.log' `
    -RedirectStandardError 'D:\RGB_SFM_artifacts\lan_proxy_error.log'
Write-Output "Artifact LAN proxy PID: $($process.Id); WSL: $wslAddress"
