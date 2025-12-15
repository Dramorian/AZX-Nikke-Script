# AZX-Nikke-Script
This is a personal Python script to solve the possible sums in the AZX Time Service minigame recently added in Nikke.
The way it works is by using hotkeys in your keyboard to get the numbers in the minigame using template matching, check sums in every direction, and it tells you both on the console and visually using a overlay where the sums are located.

This script is not user-friendly and is highly specific to my setup. It will only work out-of-the-box if you meet these conditions:
* You are playing Nikke on a **1080p monitor** in **fullscreen** mode.
* You are playing the **10x16 level**.
* You are playing with **Soline**.
Otherwise, the script will not function correctly, and you will need to modify the code and potentially the templates.

HOW THE SCRIPT WORKS:
The script is controlled by hotkeys ranging from F1 to F5.

1.  **Scanning (F5):** The script first scans your screen to identify all numbers in the generated level using template matching. It checks each position, compares it against the templates for numbers 1-9, and selects the best match. This process builds a list of lists, which acts as the game's matrix.
2.  **Checking Sums (F2, F3, F4):** Once the matrix is generated, the remaining hotkeys check for valid sums:
    * **F2:** Checks sums to the **right**.
    * **F3:** Checks sums **down**.
    * **F4:** Checks **square** sums.
3.  **Visual Output:** Results are displayed via an **always-on-top, click-through overlay** created when the script starts. Valid sums are highlighted with different colors for visual ease.
4.  **Clearing Overlay (F1):** Use **F1** to clear the overlay of all colors. This is essential after performing sums in the minigame, allowing the script to correctly check new sums that appear when numbers vanish.
5.  **Exiting:** Use **Ctrl + C** in the console (CMD/VS Code) to terminate the script.

WHAT YOU NEED:
- Python installed (3.x version)
- On your terminal install all these libraries: pip install keyboard mss numpy opencv-python PyQt5

HOW TO RUN:
1. Download the azxScript.py file and templates folder and put them in the same folder on your pc
2. Open CMD on Windows, and type cd *location of the folder where you put the azxScript file and templates* for example: cd C:\Users\mig\Desktop\AZX minigame script
3. On CMD, run the file using: python azxScript.py (if that doesn't work use py azxScript.py)
4. If you want to stop it use CTRL + C or close CMD or kill the process in the task manager.

HOW TO CORRECTLY USE IT:
When you run the script, on the terminal it shows you all the hotkeys: 
<img width="401" height="225" alt="image" src="https://github.com/user-attachments/assets/7b3d2a99-e4f2-472c-9850-df97ce57c7bd" />

When the minigame shows you all the numbers, press F5 (you might need to have CMD on focus clicking on it, I run VS Code as a administrator and don't have to focus the window). After some seconds, it will show you the generated matrix. After that, you can use F1 - F4 to manipulate the matrix / overlay. Use it slowly and one by one, because otherwise it will not show the overlay correctly because colors will overlap and stuff. 

**Recommended Workflow:**
1.  **F2** (Right sums). Complete sums in the minigame.
2.  **F1** (Clean overlay).
3.  **F3** (Down sums). Complete sums in the minigame.
4.  **F1** (Clean overlay).
5.  **F4** (Square sums). Complete sums in the minigame.
6.  **F1** (Clean overlay).
Rinse and repeat.

**Do not use F5 again** after the first press, as the script does not support re-scanning the matrix.

THINGS THAT COULD MAKE THE SCRIPT BETTER:
- Anything that makes it user friendly: better way to stop the script from running, work on different resolutions, performance improvements in the algorithm, etc.

You can do pull requests and anything that could make the script better. Like I said I made it for myself first and foremost but I liked the end result enough to make it public and of course I will welcome any improvement.

A video showing the script working:

https://github.com/user-attachments/assets/65688999-e9c6-4ca5-a9d7-6ce0f5e2dda7


