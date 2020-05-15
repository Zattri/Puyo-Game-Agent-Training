# puyo-gym-training
A repository for the training model for the machine learning platform puyo-gym

## Project Dependencies:

### Python3 (V3.7+):
Install: Google it

### pip3:
```
sudo apt-get install python3-pip

pip3 install <package>
```
- gym
- gym-retro
- numpy
- scikit-image

## Setup Local Game Env
Move `Puyo-Genesis` folder in repo to `/home/<username>/.local/lib/python3.7/site-packages/retro/data/stable`

## Controls
A = Left
S = Down
D = Right

Left = Rotate Anti-clockwise
Right = Rotate Clockwise

## Running the Training Environment:
The following commands execute one training game at a specified difficulty against the AI of the base game. Your gameplay is recorded and saved to a file in the experiences folder, with a prompt allowing you to name the file after the match concludes. Please name each match uniquely with the schema of "S\<difficulty number\>_\<match_number\>". Once you have played as many matches as desired please place the files in a folder labeled with your name inside the experiences folder.

Please record the score and time of each match (found in console after matches conclude) in a notepad file, and place in the folder alongside experience replay data.

### Basic Training Command
The match will be played on one of 5 randomised stages against the first difficulty of in-game AI, this setting is the most recommended.
```
python3 interactive.py -st random -d 1
```

### Advanced Training Command
This command starts a match at a higher difficulty, this setting is recommended for advanced players or those that wish to gather a large amount of varied data.
```
python3 interactive.py -st random -d 2
```

### Veteren Training Command
This command starts a match at the third difficulty of AI, this setting is not recommended as it is very hard.
```
python3 interactive.py -st random -d 3
```
