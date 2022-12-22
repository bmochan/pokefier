import cv2
import math
import random
import asyncio
import discord
import requests
import numpy as np
import tensorflow as tf


def catch(model: tf.keras.Sequential, pokemons: list, image_url: str, message: discord.Message, loop) -> None:
    name = identify(model, pokemons, image_url)
    name = name.lower()

    # 20% chance to incorrectly name a pokemon
    if random.randint(0, 100) <= 20:
        asyncio.run_coroutine_threadsafe(typofied_catch(message, name), loop)

    else:
        catch_string = f"<@716390085896962058> c {name}"

        if loop.is_closed() == False:
            loop.create_task(message.channel.send(catch_string))

    return


def identify(model: tf.keras.Sequential, pokemons: list, pokemon_image: str) -> str:
    original_image = cv2.imdecode(np.asarray(bytearray(requests.get(
        pokemon_image, stream=True).raw.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    image = original_image

    # resize the image to match dimensions required by the model
    img = cv2.resize(image, (200, 125))

    img = img/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred, axis=1).tolist()[0]

    pred_name = pokemons[idx]

    return pred_name


def get_keys_around(key: str) -> str:
    neighboring_keys = []

    if key.strip() == "":
        neighboring_keys = ["z", "x", "c", "v", "b", "n", "m"]

    else:
        keyboard_rows = ' qwertyuiop[]\\', ' asdfghjkl;\' ', ' zxcvbnm,./ '
        indexes = []

        rows = None

        for row_idx, row in enumerate(keyboard_rows):
            if key in row:
                key_idx = row.find(key)
                indexes.append((row_idx, key_idx))

        row_index, key_index = indexes[0]

        if row_index:
            rows = keyboard_rows[row_index-1: row_index+2]

        else:
            rows = keyboard_rows[0: 2]

        for row in rows:
            for i in [-1, 0, 1]:
                if len(row) > key_index + i and row[key_index + i] != key and key_index + i >= 0:
                    neighboring_keys.append(row[key_index + i])

    return neighboring_keys


def typofy(pokemon_name: str) -> str:
    # 10% of total characters to be typofied
    typo_count = math.ceil(len(pokemon_name) * 10/100)

    # avoid repetition
    prev_typofied_position = None
    prev_typofied_char = None

    characters = list(pokemon_name)

    iterations = 0

    while iterations < typo_count:
        iterations += 1

        typofy_position = random.randint(0, len(characters) - 1)

        char_to_typofy = characters[typofy_position]

        if typofy_position == 0:
            continue

        if typofy_position == prev_typofied_position:
            continue

        if char_to_typofy == " " and prev_typofied_char == " ":
            continue

        prev_typofied_position = typofy_position
        prev_typofied_char = char_to_typofy

        neighboring_chars = get_keys_around(char_to_typofy)
        typofied_char = neighboring_chars[random.randint(
            0, (len(neighboring_chars) - 1))]

        new_characters = []
        idx = 0

        while idx < len(characters):
            if idx == typofy_position:
                new_characters.append(typofied_char)

            else:
                new_characters.append(characters[idx])

            idx += 1

        characters = new_characters

    typofied_name = "".join(characters)

    return typofied_name


async def typofied_catch(message: discord.Message, pokemon_name: str) -> None:
    typofied = typofy(pokemon_name)
    catch_string = f"<@716390085896962058> c {typofied}"

    msg = await message.channel.send(catch_string)

    catch_string = f"<@716390085896962058> c {pokemon_name}"
    
    await asyncio.sleep(0.7)
    await msg.edit(content=catch_string)

    return
