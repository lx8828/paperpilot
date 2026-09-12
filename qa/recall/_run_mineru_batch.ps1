# 批量解析：A 桶 33 篇 arXiv PDF → out_mineru/<stem>/（已有 content_list 的跳过）
# 用途：QASPER 表格通道扩容实验的输入（QASPER 自身只给 PNG，无数值文本）
cd f:/paperpilot
$pids = @(
  '1603.00968','1605.07683','1611.03382','1701.00185','1803.07771','1804.08139','1808.06834',
  '1810.00663','1901.00439','1901.00570','1902.00330','1905.00563','1908.01060','1909.00088',
  '1909.00279','1909.13362','1909.13375','1910.02339','1911.03243','1911.11933','2001.00137',
  '2001.05467','2001.08845','2001.10179','2002.00876','2002.02224','2002.05058','2002.06675',
  '2002.08307','2002.10361','2002.11402','2003.09520','2004.03061'
)
$mineru = "f:/paperpilot/.venv-mineru/Scripts/mineru.exe"
foreach ($p in $pids) {
  $pdf = "assets/papers/$($p)v1.pdf"
  $out = "assets/artifacts/out_mineru/$($p)v1"
  if (Test-Path "$out/auto/$($p)v1_content_list.json") {
    Write-Output "[skip] $p (已有产物)"
    continue
  }
  if (-not (Test-Path $pdf)) { Write-Output "[miss] $p 缺 PDF"; continue }
  $t0 = Get-Date
  & $mineru -p $pdf -o $out -b pipeline *> "qa/recall/_mineru_$($p).log"
  $ok = Test-Path "$out/auto/$($p)v1_content_list.json"
  Write-Output ("[{0}] {1} in {2:N0}s" -f ($(if($ok){'ok'}else{'FAIL'})), $p, ((Get-Date)-$t0).TotalSeconds)
}
Write-Output "ALL DONE"
