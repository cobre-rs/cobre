# Cobre Validate Demo
# Generates: validation.gif

Require cobre

Set Shell "bash"
Set FontFamily "JetBrains Mono"
Set FontSize 16
Set Width 1200
Set Height 600
Set Background "#0F1419"
Set Foreground "#C8C6C2"
Set CursorColor "#B87333"
Set SelectionColor "#1A2028"
Set WindowBar "Colorful"
Set BorderRadius 8
Set Padding 12

Output validation.gif

Type "rm -rf demo/"
Enter
Sleep 500ms

Type "cobre init --template 1dtoy demo/"
Enter
Sleep 2s

Type "cobre validate demo/"
Enter
Sleep 5s

Sleep 2s
