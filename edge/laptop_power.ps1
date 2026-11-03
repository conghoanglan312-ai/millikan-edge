param([string]$Out, [int]$SettleS = 30, [int]$IdleS = 120, [double]$Minutes = 10, [string]$WaitGone = "")
# Laptop power: waits for the charger to come out, then logs the battery's own
# discharge rate every 5 s through a settle, an idle and a soak.py phase.
$repo = "C:\Users\LAN\millikan-edge"
Add-Type -Namespace W -Name K -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][W.K]::SetThreadExecutionState(0x80000003)   # keep the display on, no dimming
"t_s,phase,onAC,dischargeRate_mW,remaining_mWh,voltage_mV,brightness" | Add-Content $Out
function Sample($phase) {
    $b = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus
    $l = (Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightness).CurrentBrightness
    $s = ((Get-Date) - $script:t0).TotalSeconds
    "{0:F1},{1},{2},{3},{4},{5},{6}" -f $s, $phase, $b.PowerOnline, $b.DischargeRate,
        $b.RemainingCapacity, $b.Voltage, $l | Add-Content $Out
}
while ((Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus).PowerOnline) { Start-Sleep -Seconds 2 }
while ($WaitGone -and (Test-Path $WaitGone)) { Start-Sleep -Seconds 2 }   # a pulled USB stick is the go signal
$script:t0 = Get-Date
foreach ($ph in @(@("settle", $SettleS), @("idle", $IdleS))) {
    $end = (Get-Date).AddSeconds($ph[1])
    while ((Get-Date) -lt $end) { Sample $ph[0]; Start-Sleep -Seconds 5 }
}
$p = Start-Process python -ArgumentList "edge/soak.py", "--minutes", $Minutes, "--gate", "fp32" `
    -WorkingDirectory $repo -PassThru -RedirectStandardOutput "$Out.soak.txt" -RedirectStandardError "$Out.soak.err"
while (-not $p.HasExited) { Sample "soak"; Start-Sleep -Seconds 5 }
Sample "end"
[void][W.K]::SetThreadExecutionState(0x80000000)
