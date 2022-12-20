# put your token inside quotes
TOKEN = ""

# replace with id/s of server/s you want to autocatch on
WHITELISTED_SERVERS = [69696969696, 420420420420]

# replace with id/s of channel/s you don't want to autocatch on
BLACKLISTED_CHANNELS = [4206942069]

# library imports
import discord
import threading
import asyncio

# module imports
from src.utils import *
from src.captcha_solver import solve

# load the model (contact me to get the file)
model = tf.keras.models.load_model("model_pokemon.h5")

# initialize the labels
pokemons = ['10% Zygarde', 'Abomasnow', 'Abra', 'Absol', 'Accelgor', 'Aegislash', 'Aerodactyl', 'Aggron', 'Aipom', 'Alakazam', 'Alcremie', 'Alolan Diglett', 'Alolan Dugtrio', 'Alolan Geodude', 'Alolan Golem', 'Alolan Graveler', 'Alolan Grimer', 'Alolan Marowak', 'Alolan Meowth', 'Alolan Ninetales', 'Alolan Persian', 'Alolan Raichu', 'Alolan Rattata', 'Alolan Sandshrew', 'Alolan Sandslash', 'Alolan Vulpix', 'Alomomola', 'Altaria', 'Amaura', 'Ambipom', 'Amoonguss', 'Ampharos', 'Anorith', 'Appletun', 'Applin', 'Araquanid', 'Arbok', 'Arcanine', 'Arceus', 'Archen', 'Archeops', 'Arctovish', 'Arctozolt', 'Ariados', 'Armaldo', 'Aromatisse', 'Aron', 'Arrokuda', 'Articuno', 'Attack Deoxys', 'Audino', 'Aurorus', 'Autumn Chikorita', 'Autumn Pansage', 'Autumn Rapidash', 'Autumn Skiddo', 'Autumn Snivy', 'Avalugg', 'Axew', 'Azelf', 'Azumarill', 'Azurill', 'Bagon', 'Baltoy', 'Banette', 'Barbaracle', 'Barboach', 'Barraskewda', 'Basculegion', 'Basculin', 'Bastiodon', 'Bayleef', 'Beartic', 'Beautifly', 'Beedrill', 'Beheeyem', 'Beldum', 'Bellossom', 'Bellsprout', 'Bergmite', 'Bewear', 'Bibarel', 'Bidoof', 'Binacle', 'Bisharp', 'Blastoise', 'Blaziken', 'Blipbug', 'Blissey', 'Blitzle', 'Blue-Striped Basculin', 'Boldore', 'Boltund', 'Bonsly', 'Bouffalant', 'Bounsweet', 'Braixen', 'Braviary', 'Breloom', 'Brionne', 'Bronzong', 'Bronzor', 'Bruxish', 'Budew', 'Buizel', 'Bulbasaur', 'Buneary', 'Bunnelby', 'Burmy', 'Butterfree', 'Cacnea', 'Cacturne', 'Camerupt', 'Carbink', 'Carkol', 'Carnivine', 'Carracosta', 'Carvanha', 'Cascoon', 'Castform', 'Caterpie', 'Celebi', 'Celesteela', 'Centiskorch', 'Chandelure', 'Chansey', 'Charizard', 'Charjabug', 'Charmander', 'Charmeleon', 'Chatot', 'Cherrim', 'Cherubi', 'Chesnaught', 'Chespin', 'Chewtle', 'Chikorita', 'Chimchar', 'Chimecho', 'Chinchou', 'Chingling', 'Cinccino', 'Cinderace', 'Clamperl', 'Clauncher', 'Clawitzer', 'Claydol', 'Clefable', 'Clefairy', 'Cleffa', 'Clobbopus', 'Cloyster', 'Coalossal', 'Cobalion', 'Cofagrigus', 'Combee', 'Combusken', 'Comfey', 'Conkeldurr', 'Copperajah', 'Corphish', 'Corsola', 'Corviknight', 'Corvisquire', 'Cosmoem', 'Cosmog', 'Cottonee', 'Crabominable', 'Crabrawler', 'Cradily', 'Cramorant', 'Cranidos', 'Crawdaunt', 'Cresselia', 'Croagunk', 'Crobat', 'Croconaw', 'Crustle', 'Cryogonal', 'Cubchoo', 'Cubone', 'Cufant', 'Cursola', 'Cutiefly', 'Cyndaquil', 'Darkrai', 'Darmanitan', 'Dartrix', 'Darumaka', 'Decidueye', 'Dedenne', 'Deerling', 'Defense Deoxys', 'Deino', 'Delcatty', 'Delibird', 'Delphox', 'Dewgong', 'Dewott', 'Dewpider', 'Dhelmise', 'Dialga', 'Diancie', 'Diggersby', 'Diglett', 'Ditto', 'Dodrio', 'Doduo', 'Donphan', 'Dottler', 'Doublade', 'Dracovish', 'Dracozolt', 'Dragalge', 'Dragapult', 'Dragonair', 'Dragonite', 'Drakloak', 'Drampa', 'Drapion', 'Dratini', 'Drednaw', 'Dreepy', 'Drifblim', 'Drifloon', 'Drilbur', 'Drizzile', 'Drowzee', 'Druddigon', 'Dubwool', 'Ducklett', 'Dugtrio', 'Dunsparce', 'Duosion', 'Duraludon', 'Durant', 'Dusclops', 'Dusknoir', 'Duskull', 'Dustox', 'Dwebble', 'Eelektrik', 'Eelektross', 'Eevee', 'Eiscue', 'Ekans', 'Eldegoss', 'Electabuzz', 'Electivire', 'Electrike', 'Electrode', 'Elekid', 'Elgyem', 'Emboar', 'Emolga', 'Empoleon', 'Entei', 'Escavalier', 'Espeon', 'Espurr', 'Excadrill', 'Exeggcute', 'Exeggutor', 'Exploud', 'Falinks', 'Farfetchd', 'Fearow', 'Feebas', 'Fennekin', 'Feraligatr', 'Ferroseed', 'Ferrothorn', 'Finneon', 'Flaaffy', 'Flabebe', 'Flapple', 'Flareon', 'Fletchinder', 'Fletchling', 'Floatzel', 'Floette', 'Florges', 'Flygon', 'Fomantis', 'Foongus', 'Forretress', 'Fraxure', 'Frillish', 'Froakie', 'Frogadier', 'Froslass', 'Frosmoth', 'Furfrou', 'Furret', 'Gabite', 'Galarian Corsola', 'Galarian Darmanitan', 'Galarian Farfetchd', 'Galarian Linoone', 'Galarian Meowth', 'Galarian Moltres', 'Galarian Mr. Mime', 'Galarian Ponyta', 'Galarian Rapidash', 'Galarian Slowking', 'Galarian Slowpoke', 'Galarian Stunfisk', 'Galarian Weezing', 'Galarian Yamask', 'Galarian Zapdos', 'Galarian Zigzagoon', 'Gallade', 'Galvantula', 'Garbodor', 'Garchomp', 'Gardevoir', 'Gastly', 'Gastrodon', 'Genesect', 'Gengar', 'Geodude', 'Gible', 'Gigalith', 'Girafarig', 'Glaceon', 'Glalie', 'Glameow', 'Gligar', 'Gliscor', 'Gloom', 'Gogoat', 'Golbat', 'Goldeen', 'Golduck', 'Golem', 'Golett', 'Golisopod', 'Golurk', 'Goodra', 'Goomy', 'Gorebyss', 'Gossifleur', 'Gothita', 'Gothitelle', 'Gothorita', 'Gourgeist', 'Granbull', 'Grapploct', 'Graveler', 'Greedent', 'Greninja', 'Grimer', 'Grimmsnarl', 'Grookey', 'Grotle', 'Groudon', 'Grovyle', 'Growlithe', 'Grubbin', 'Grumpig', 'Gulpin', 'Gumshoos', 'Gurdurr', 'Gyarados', 'Hakamo-o', 'Happiny', 'Hariyama', 'Hatenna', 'Hatterene', 'Hattrem', 'Haunter', 'Hawlucha', 'Haxorus', 'Heatmor', 'Heatran', 'Heliolisk', 'Helioptile', 'Heracross', 'Herdier', 'Hippopotas', 'Hippowdon', 'Hisuian Avalugg', 'Hisuian Braviary', 'Hisuian Goodra', 'Hisuian Growlithe', 'Hisuian Lilligant', 'Hisuian Qwilfish', 'Hisuian Samurott', 'Hisuian Sliggoo', 'Hisuian Sneasel', 'Hisuian Typhlosion', 'Hisuian Voltorb', 'Hisuian Zoroark', 'Hisuian Zorua', 'Hitmonchan', 'Hitmonlee', 'Hitmontop', 'Ho-Oh', 'Honchkrow', 'Honedge', 'Hoopa', 'Hoothoot', 'Hoppip', 'Horsea', 'Houndoom', 'Houndour', 'Huntail', 'Hydreigon', 'Hypno', 'Igglybuff', 'Illumise', 'Impidimp', 'Incineroar', 'Indeedee', 'Infernape', 'Inkay', 'Inteleon', 'Ivysaur', 'Jangmo-o', 'Jellicent', 'Jigglypuff', 'Jirachi', 'Jolteon', 'Joltik', 'Jumpluff', 'Jynx', 'Kabuto', 'Kabutops', 'Kadabra', 'Kakuna', 'Kangaskhan', 'Karrablast', 'Kecleon', 'Keldeo', 'Kingdra', 'Kingler', 'Kirlia', 'Klang', 'Kleavor', 'Klefki', 'Klink', 'Klinklang', 'Koffing', 'Komala', 'Kommo-o', 'Krabby', 'Kricketot', 'Kricketune', 'Krokorok', 'Krookodile', 'Kubfu', 'Kyurem',
            'Lairon', 'Lampent', 'Landorus', 'Lanturn', 'Lapras', 'Larvesta', 'Larvitar', 'Leafeon', 'Leavanny', 'Ledian', 'Ledyba', 'Lickilicky', 'Lickitung', 'Liepard', 'Lileep', 'Lilligant', 'Lillipup', 'Linoone', 'Litleo', 'Litten', 'Litwick', 'Lombre', 'Lopunny', 'Lotad', 'Loudred', 'Lucario', 'Ludicolo', 'Lumineon', 'Lunala', 'Lunatone', 'Lurantis', 'Luvdisc', 'Luxio', 'Luxray', 'Lycanroc', 'Machamp', 'Machoke', 'Machop', 'Magby', 'Magcargo', 'Magearna', 'Magikarp', 'Magmar', 'Magmortar', 'Magnemite', 'Magneton', 'Magnezone', 'Makuhita', 'Malamar', 'Mamoswine', 'Manaphy', 'Mandibuzz', 'Manectric', 'Mankey', 'Mantine', 'Mantyke', 'Maractus', 'Mareanie', 'Mareep', 'Marill', 'Marowak', 'Marshadow', 'Marshtomp', 'Masquerain', 'Mawile', 'Medicham', 'Meditite', 'Meganium', 'Melmetal', 'Meloetta', 'Meltan', 'Meowstic', 'Meowth', 'Mesprit', 'Metagross', 'Metang', 'Metapod', 'Mew', 'Mewtwo', 'Mienfoo', 'Mienshao', 'Mightyena', 'Milcery', 'Milotic', 'Miltank', 'Mime Jr', 'Mimikyu', 'Minccino', 'Minior', 'Minun', 'Misdreavus', 'Mismagius', 'Moltres', 'Monferno', 'Morelull', 'Morgrem', 'Morpeko', 'Mothim', 'Mr. Mime', 'Mr. Rime', 'Mudbray', 'Mudkip', 'Mudsdale', 'Muk', 'Munchlax', 'Munna', 'Murkrow', 'Musharna', 'Natu', 'Necrozma', 'Nickit', 'Nidoking', 'Nidoqueen', 'NidoranF', 'NidoranM', 'Nidorina', 'Nidorino', 'Nincada', 'Ninetales', 'Ninjask', 'Noctowl', 'Noibat', 'Noivern', 'Nosepass', 'Numel', 'Nuzleaf', 'Obstagoon', 'Octillery', 'Oddish', 'Omanyte', 'Omastar', 'Onix', 'Oranguru', 'Orbeetle', 'Oricorio', 'Oshawott', 'Overqwil', 'Pachirisu', 'Palkia', 'Palossand', 'Palpitoad', 'Pancham', 'Pangoro', 'Panpour', 'Pansage', 'Pansear', 'Paras', 'Parasect', 'Passimian', 'Patrat', 'Oricorio', 'Pawniard', 'Pelipper', 'Perrserker', 'Persian', 'Petilil', 'Phanpy', 'Phantump', 'Phione', 'Pichu', 'Pidgeot', 'Pidgeotto', 'Pidgey', 'Pidove', 'Pignite', 'Pikachu', 'Pikipek', 'Piloswine', 'Pincurchin', 'Pineco', 'Pinsir', 'Piplup', 'Plusle', 'Politoed', 'Poliwag', 'Poliwhirl', 'Poliwrath', 'Polteageist', 'Pom-pom Oricorio', 'Ponyta', 'Poochyena', 'Popplio', 'Porygon', 'Porygon-Z', 'Porygon2', 'Primarina', 'Primeape', 'Prinplup', 'Probopass', 'Psyduck', 'Pumpkaboo', 'Pupitar', 'Purrloin', 'Purugly', 'Pyroar', 'Pyukumuku', 'Quagsire', 'Quilava', 'Quilladin', 'Qwilfish', 'Raboot', 'Raichu', 'Rainy Castform', 'Ralts', 'Rampardos', 'Rapid Strike Urshifu', 'Rapidash', 'Raticate', 'Rattata', 'Rayquaza', 'Regieleki', 'Regigigas', 'Registeel', 'Relicanth', 'Remoraid', 'Reshiram', 'Reuniclus', 'Rhydon', 'Rhyhorn', 'Rhyperior', 'Ribombee', 'Rillaboom', 'Riolu', 'Rockruff', 'Roggenrola', 'Rolycoly', 'Rookidee', 'Roselia', 'Roserade', 'Rotom', 'Rowlet', 'Rufflet', 'Runerigus', 'Sableye', 'Salamence', 'Salandit', 'Salazzle', 'Samurott', 'Sandaconda', 'Sandile', 'Sandshrew', 'Sandslash', 'Sandy Wormadam', 'Sandygast', 'Sawk', 'Sawsbuck', 'Scatterbug', 'Sceptile', 'Scizor', 'Scolipede', 'Scorbunny', 'Scrafty', 'Scraggy', 'Scyther', 'Seadra', 'Seaking', 'Sealeo', 'Seedot', 'Seel', 'Seismitoad', 'Sensu Oricorio', 'Sentret', 'Serperior', 'Servine', 'Seviper', 'Sewaddle', 'Sharpedo', 'Shaymin', 'Shedinja', 'Shelgon', 'Shellder', 'Shellos', 'Shelmet', 'Shieldon', 'Shiftry', 'Shiinotic', 'Shinx', 'Shroomish', 'Shuckle', 'Shuppet', 'Sigilyph', 'Silcoon', 'Silicobra', 'Silvally', 'Simipour', 'Simisage', 'Simisear', 'Sinistea', 'Sirfetchd', 'Sizzlipede', 'Skarmory', 'Skiddo', 'Skiploom', 'Skitty', 'Skorupi', 'Skrelp', 'Skuntank', 'Skwovet', 'Slaking', 'Slakoth', 'Sliggoo', 'Slowbro', 'Slowking', 'Slowpoke', 'Slugma', 'Slurpuff', 'Smeargle', 'Smoochum', 'Sneasel', 'Sneasler', 'Snivy', 'Snom', 'Snorlax', 'Snorunt', 'Snover', 'Snowy Castform', 'Snubbull', 'Sobble', 'Solgaleo', 'Solosis', 'Solrock', 'Spearow', 'Spectrier', 'Spewpa', 'Spheal', 'Spinarak', 'Spinda', 'Spiritomb', 'Spoink', 'Spritzee', 'Squirtle', 'Stakataka', 'Stantler', 'Staraptor', 'Staravia', 'Starly', 'Starmie', 'Staryu', 'Steelix', 'Steenee', 'Stonjourner', 'Stoutland', 'Stufful', 'Stunfisk', 'Stunky', 'Sudowoodo', 'Suicune', 'Sunflora', 'Sunkern', 'Sunny Castform', 'Surskit', 'Swablu', 'Swadloon', 'Swalot', 'Swampert', 'Swanna', 'Swellow', 'Swinub', 'Swirlix', 'Swoobat', 'Sylveon', 'Taillow', 'Talonflame', 'Tangela', 'Tangrowth', 'Tapu Bulu', 'Tapu Fini', 'Tapu Lele', 'Tauros', 'Teddiursa', 'Tentacool', 'Tentacruel', 'Tepig', 'Terrakion', 'Thievul', 'Throh', 'Thwackey', 'Timburr', 'Tirtouga', 'Togedemaru', 'Togekiss', 'Togepi', 'Togetic', 'Torchic', 'Torkoal', 'Tornadus', 'Torracat', 'Torterra', 'Totodile', 'Toucannon', 'Toxapex', 'Toxel', 'Toxicroak', 'Toxtricity', 'Tranquill', 'Trapinch', 'Trash Wormadam', 'Treecko', 'Trevenant', 'Tropius', 'Trubbish', 'Trumbeak', 'Tsareena', 'Turtonator', 'Turtwig', 'Tympole', 'Tynamo', 'Typhlosion', 'Tyranitar', 'Tyrantrum', 'Tyrogue', 'Tyrunt', 'Umbreon', 'Unfezant', 'Unown', 'Ursaluna', 'Ursaring', 'Urshifu', 'Vanillish', 'Vanillite', 'Vanilluxe', 'Vaporeon', 'Venipede', 'Venomoth', 'Venonat', 'Venusaur', 'Vespiquen', 'Vibrava', 'Victini', 'Victreebel', 'Vigoroth', 'Vikavolt', 'Vileplume', 'Virizion', 'Vivillon', 'Volbeat', 'Volcanion', 'Volcarona', 'Voltorb', 'Vullaby', 'Vulpix', 'Wailmer', 'Wailord', 'Walrein', 'Wartortle', 'Watchog', 'Weavile', 'Weedle', 'Weepinbell', 'Weezing', 'Whimsicott', 'Whirlipede', 'Whiscash', 'Whismur', 'Wigglytuff', 'Wimpod', 'Wingull', 'Wishiwashi', 'Wobbuffet', 'Woobat', 'Wooloo', 'Wooper', 'Wormadam', 'Wurmple', 'Wynaut', 'Wyrdeer', 'Xatu', 'Xerneas', 'Xurkitree', 'Yamask', 'Yamper', 'Yanma', 'Yanmega', 'Yungoos', 'Yveltal', 'Zacian', 'Zangoose', 'Zapdos', 'Zarude', 'Zebstrika', 'Zekrom', 'Zeraora', 'Zigzagoon', 'Zoroark', 'Zorua', 'Zubat', 'Zweilous', 'Zygarde']

# kickstart the model to reduce prediction time
identify(model, pokemons,
         "https://media.discordapp.net/attachments/1037799258637750363/1037814779005382716/pokemon.jpg")


class Pokefier(discord.Client):
    async def on_ready(self):
        self.verifying = False
        await self.change_presence(status=discord.Status.dnd)
        print('[READY] Logged in as', self.user)

    async def on_message(self, message):
        if not hasattr(self, 'verifying'):
            self.verifying = False

        # only acknowledge the bot's messages if captcha is not being solved
        if message.author.id == 716390085896962058 and self.verifying == False:
            if len(message.embeds) > 0 and "wild pokémon has appeared!" in message.embeds[0].title:
                if message.guild.id in WHITELISTED_SERVERS and message.channel.id not in BLACKLISTED_CHANNELS:
                    pokemon_image = message.embeds[0].image.url

                    await message.channel.trigger_typing()

                    threading.Thread(target=catch, args=(
                        model, pokemons, pokemon_image, message, asyncio.get_event_loop())).start()

            if message.content.startswith("Whoa there") and str(self.user.id) in message.content:
                self.verifying = True

                # getting the url from captcha message
                url = message.content.split(
                    "Whoa there. Please tell us you're human! ")[1]

                print("Got CAPTCHA! Attempting to solve.")

                self.captcha_url = url

                threading.Thread(target=solve, args=(
                    client, self.captcha_url)).start()


client = Pokefier(
    guild_subscription_options=discord.GuildSubscriptionOptions.off())
client.run(TOKEN)
