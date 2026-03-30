# Cobre Multi-threading Speedup Demo
# Generates: multithreading.gif
# Shows the wall-clock time difference between --threads 1 and --threads 4.

Require cobre

Set Shell "bash"
Set FontFamily "JetBrains Mono"
Set FontSize 14
Set Width 1200
Set Height 600
Set Theme { "background": "#0F1419", "foreground": "#C8C6C2", "cursor": "#B87333", "selection": "#1A2028", "black": "#0F1419", "red": "#DC4C4C", "green": "#4A8B6F", "yellow": "#F5A623", "blue": "#4A90B8", "magenta": "#D4956A", "cyan": "#4A8B6F", "white": "#C8C6C2", "brightBlack": "#2D3440", "brightRed": "#DC4C4C", "brightGreen": "#4A8B6F", "brightYellow": "#F5A623", "brightBlue": "#4A90B8", "brightMagenta": "#D4956A", "brightCyan": "#4A8B6F", "brightWhite": "#E8E6E3" }
Set WindowBar Colorful
Set BorderRadius 6
Set Padding 5
Set Margin 3
Set MarginFill "#2D3440"

Output multithreading.gif

Type "cobre init --template 1dtoy demo/"
Enter
Sleep 4s

# Increase forward passes to 8 and simulation scenarios to 1000 for a longer run
Type "jq '.training.forward_passes = 8 | .simulation.num_scenarios = 1000' demo/config.json > tmp.json && mv tmp.json demo/config.json"
Enter
Sleep 4s

Type "cobre run demo/ --threads 1"
Enter
Sleep 30s

Type "cobre run demo/ --threads 4"
Enter
Sleep 15s

Sleep 2s
