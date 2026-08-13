$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
python -m pip install `
  "$Root\engines\isgn" `
  "$Root\engines\m2n" `
  "$Root\engines\fec" `
  "$Root"
dor stack-info
