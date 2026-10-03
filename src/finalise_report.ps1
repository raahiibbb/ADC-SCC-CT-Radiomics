# Update the table of contents in Word, save, then export a PDF.
# Two separate Word sessions with a pause between them: saving and exporting in one
# session, or starting the second session too quickly, hangs Word on this document.
# Usage (in PowerShell):  & .\src\finalise_report.ps1
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$f = Join-Path $root "report\Group_13_EEE_402_G2_Project_Report"

function Close-Word($w) {
    $w.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($w)
    [GC]::Collect(); [GC]::WaitForPendingFinalizers()
    Start-Sleep -Seconds 3
}

$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0
$d = $w.Documents.Open("$f.docx", $false, $false, $false)
$d.TablesOfContents(1).Update(); $d.Save(); $d.Close($false)
Close-Word $w
"toc updated"

$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0
$d = $w.Documents.Open("$f.docx", $false, $true, $false)
$d.ExportAsFixedFormat("${f}_new.pdf", 17)      # a PDF open in a viewer cannot be overwritten directly
"pages: " + $d.ComputeStatistics(2)
$d.Close($false)
Close-Word $w
Move-Item -Force "${f}_new.pdf" "$f.pdf"
"saved $f.pdf"
