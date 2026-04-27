@echo off
set TCL_LIBRARY=C:\Users\zachp\AppData\Local\Programs\Python\Python313\tcl\tcl8.6
set TK_LIBRARY=C:\Users\zachp\AppData\Local\Programs\Python\Python313\tcl\tk8.6
cd /d C:\Users\zachp\RaidVision
call venv\Scripts\activate.bat
python gui.py
