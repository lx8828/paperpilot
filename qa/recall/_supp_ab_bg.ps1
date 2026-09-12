cd C:\Users\83465\Desktop\paperpilot
uv run python qa/recall/_supp_ab.py 2>&1 | Out-File -FilePath qa\recall\_supp_ab.log -Encoding utf8
Write-Host "all done"
