# Pokéfier
The best autocatching program on Pokétwo to have been created till date. Why? Because it is free, automatically solves CAPTCHAs and doesn't log your token.

## Setup 
- Download Python version 3.10.8 if you haven't already. Click [here](https://www.python.org/downloads/release/python-3108/) to do this.
- Run the Python installer and make sure you add python to PATH. Click on 'Install Now' and wait for the installation to finish.
- Download this repository or git clone it.
- Double click or `cd` into `pokefier/autocatcher` then open a terminal. Input the command `pip install -r requirements.txt` and wait till it's complete.
- Download ffmpeg from [here](https://github.com/BtbN/FFmpeg-Builds/releases) and unzip the folder inside to a location.
- Add the location of the folder to PATH. Refer to [this](https://www.youtube.com/watch?v=gb9e3m98avk) video and follow the steps to achieve this.
- After completion of the steps above, open the `main.py` file using a text editor (e.g notepad) and put the values of `WHITELISTED_SERVERS` and `TOKEN` as instructed inside.
- Finally, open a terminal and run `py main.py`. The program will automatically catch a pokemon once it spawns in any of the `WHITELISTED_SERVERS`.

## Background
I was looking for a Pokétwo autocatcher on Github for quite a while and the ones I could find were either paid or extremely sketchy. There were some free ones which used the infamous "hint exploit" but that was just very unprofessional to me. Paying to exploit a Discord bot or running obfuscated code that can do god-knows-what was just not acceptable. So, if you are like me and can relate, Pokéfier is made just for you.

## Pokéfier and Machine Learning
Pokéfier uses machine learning to identify pokemon images, given a url or an image object. The program and the model is completely free. However, you would need to join our Discord server to avail the `model`.

The creation of this project was just like any other Machine Learning project. To recreate it, you would have to:
1. Collect data
2. Prepare the dataset (referred to as data cleaning)
3. Create a model 
4. Train the model
5. Make predictions using the model

You can do this using a programming language of your choice. In all of this, collecting the data will be the toughest part. I have made a program to collect Pokémon images and label them accordingly, which I plan to open-source if this repository gets 500 stars.

## Plans for the future
This project is still a WIP and I will update it frequently when I get time. Others are encouraged to collaborate and find ways to improve the program. Contact me at iambmo#1442 on Discord.
