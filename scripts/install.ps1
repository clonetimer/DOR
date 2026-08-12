$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
python -m pip install `
  "$Root\wheels\isgn_toolkit-2.2.0-py3-none-any.whl" `
  "$Root\wheels\m2n_toolkit-2.3.1-py3-none-any.whl" `
  "$Root\wheels\fec_toolkit-2.4.0-py3-none-any.whl" `
  "$Root\wheels\dor_diagnostic_stack-0.6.0-py3-none-any.whl"
dor stack-info
