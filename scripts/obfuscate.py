"""
obfuscate.py — Скрипт обфускации базы знаний QuantumForge RAG-бота.

Архитектурный выбор:
- Замена терминов выполняется в порядке убывания длины ключа (longest-match-first),
  чтобы составные фразы ("Darth Vader") заменялись раньше одиночных слов ("Vader").
- Используется re.sub с флагом IGNORECASE + callback для сохранения регистра первой буквы.
- Встроенные тексты (RAW_ARTICLES) содержат 30+ статей по вселенной Star Wars,
  написанных вручную в стиле энциклопедии — это гарантирует, что LLM не «вспомнит»
  их дословно, а будет вынуждена обращаться к векторному индексу.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TERMS_MAP_PATH = SCRIPT_DIR / "terms_map.json"
OUTPUT_DIR = PROJECT_ROOT / "knowledge_base"

# ---------------------------------------------------------------------------
# Встроенные статьи (30 штук) — оригинальные термины Star Wars
# ---------------------------------------------------------------------------
RAW_ARTICLES: dict[str, str] = {
    "luke_skywalker.md": """# Luke Skywalker

Luke Skywalker was a legendary Jedi Master who played a pivotal role in the fall of the Galactic Empire.
Born on Tatooine, he was the son of Anakin Skywalker and Padme Amidala.
Raised by his uncle Owen and aunt Beru on Tatooine, Luke grew up far from the conflict that consumed the galaxy.

His journey began when he encountered the droid R2-D2, who carried a message from Princess Leia.
Guided by Obi-Wan Kenobi, Luke learned the ways of the Force and joined the Rebel Alliance.
He piloted an X-Wing during the Battle of Yavin, destroying the Death Star with a precise proton torpedo shot.

Luke later trained under the Jedi Master Yoda on the swamp planet Dagobah.
He confronted Darth Vader — his own father — and ultimately redeemed him, bringing balance to the Force.
After the fall of the Galactic Empire, Luke attempted to rebuild the Jedi Order.
He wielded a green lightsaber and was known for his compassion and connection to the light side of the Force.
""",

    "darth_vader.md": """# Darth Vader

Darth Vader, born Anakin Skywalker, was a Sith Lord and the Supreme Commander of the Galactic Empire's military forces.
Once a promising Jedi Knight, he fell to the dark side of the Force after being seduced by Emperor Palpatine.

Anakin Skywalker was born on Tatooine and discovered to have an extraordinarily high midi-chlorian count.
He was trained by Obi-Wan Kenobi and became one of the most powerful Jedi in the Galactic Republic.
His fear of losing his wife Padme Amidala drove him to seek power beyond the Jedi Order's teachings.

After his transformation into Darth Vader, he wore a life-sustaining black armor suit.
He commanded Void Cruisers and led legions of Stormtroopers across the galaxy.
His signature weapon was a red lightsaber, and he was infamous for his use of Force choke on subordinates.

Darth Vader was redeemed by his son Luke Skywalker during the Battle of Endor,
turning against Emperor Palpatine and destroying him at the cost of his own life.
""",

    "princess_leia.md": """# Princess Leia Organa

Princess Leia Organa was a leader of the Rebel Alliance and one of the most prominent figures in the fight against the Galactic Empire.
She was the adopted daughter of Senator Bail Organa of Alderaan and the biological daughter of Anakin Skywalker and Padme Amidala.

Leia was a skilled diplomat and military strategist. She served in the Galactic Republic's Senate before the Empire dissolved it.
When Alderaan was destroyed by the Death Star on the orders of Grand Moff Tarkin, she became even more determined to defeat the Empire.

She was captured by Darth Vader while attempting to deliver the Death Star plans to the Rebel Alliance.
The plans were hidden inside the droid R2-D2, who eventually reached Luke Skywalker and Obi-Wan Kenobi.

Leia was also Force-sensitive, a fact revealed later in her life. She trained briefly as a Jedi under Luke Skywalker.
She led the Resistance against the First Order and was known as General Leia Organa in her later years.
""",

    "han_solo.md": """# Han Solo

Han Solo was a legendary smuggler, pilot, and general who became one of the key figures in the Rebel Alliance's victory over the Galactic Empire.
He was the captain of the Millennium Falcon, the fastest ship in the galaxy, and the partner of the Wookiee Chewbacca.

Han Solo grew up on the streets of Corellia and later served briefly in the Imperial Navy before deserting.
He became a smuggler working for Jabba the Hutt, transporting illegal cargo across the galaxy.

His life changed when he agreed to transport Luke Skywalker and Obi-Wan Kenobi to Alderaan.
Initially motivated by money, Han eventually joined the Rebel Alliance out of loyalty and friendship.
He piloted the Millennium Falcon during the Battle of Yavin, clearing the way for Luke Skywalker to destroy the Death Star.

Han Solo was frozen in carbonite by Darth Vader on Bespin and delivered to Jabba the Hutt,
but was later rescued by Princess Leia, Luke Skywalker, and Lando Calrissian.
He married Princess Leia and their son Ben Solo later became Kylo Ren.
""",

    "obi_wan_kenobi.md": """# Obi-Wan Kenobi

Obi-Wan Kenobi was a legendary Jedi Master who served the Galactic Republic and later guided Luke Skywalker in the ways of the Force.
He was the Padawan of Qui-Gon Jinn and later the master of Anakin Skywalker.

Born on Stewjon, Obi-Wan trained at the Jedi Temple on Coruscant.
He fought in the Clone Wars alongside Anakin Skywalker, commanding Replicant Legions against the Separatists.
He defeated Darth Maul on Naboo, avenging his master Qui-Gon Jinn.

After Anakin's fall to the dark side, Obi-Wan fought and defeated him on Mustafar,
leaving him gravely injured. He then went into exile on Tatooine to watch over the young Luke Skywalker.

Obi-Wan wielded a blue lightsaber and was known for his mastery of the Soresu defensive form.
He sacrificed himself during a confrontation with Darth Vader aboard the Death Star,
becoming one with the Force and continuing to guide Luke as a Force ghost.
""",

    "yoda.md": """# Yoda

Yoda was one of the most legendary and powerful Jedi Masters in the history of the Jedi Order.
He served as Grand Master of the Jedi Order and was a member of the Galactic Republic's Jedi High Council.

Yoda belonged to an unknown species and lived for approximately 900 years.
He trained generations of Jedi, including Mace Windu, Qui-Gan Jinn, and Count Dooku before his fall.
During the Clone Wars, Yoda commanded Replicant Legions and fought on multiple fronts against the Separatists.

After the execution of Order 66 and the fall of the Galactic Republic, Yoda went into exile on the swamp planet Dagobah.
There he trained Luke Skywalker in the ways of the Force, teaching him to lift rocks with the Force
and warning him about the dangers of the dark side.

Yoda wielded a green lightsaber and was known for his acrobatic fighting style despite his small stature.
He passed away peacefully on Dagobah, becoming one with the Force at the age of 900.
""",

    "emperor_palpatine.md": """# Emperor Palpatine

Emperor Palpatine, also known as Darth Sidious, was the Dark Lord of the Sith who orchestrated the fall of the Galactic Republic
and the rise of the Galactic Empire. He was the most powerful Sith Lord in millennia.

Palpatine was born on Naboo and served as its Senator in the Galactic Republic's Senate.
He secretly trained as a Sith under Darth Plagueis while publicly presenting himself as a mild-mannered politician.
He manipulated the Trade Federation into blockading Naboo, engineering a crisis that propelled him to Supreme Chancellor.

As Supreme Chancellor, Palpatine secretly orchestrated the Clone Wars, controlling both sides of the conflict.
He turned Anakin Skywalker to the dark side, creating Darth Vader, and issued Order 66 to eliminate the Jedi Order.
He then declared himself Emperor, transforming the Galactic Republic into the Galactic Empire.

Palpatine was a master of Force lightning and Sith sorcery. He was ultimately destroyed by Darth Vader
during the Battle of Endor, though he later returned through cloning on the hidden Sith world of Exegol.
""",

    "chewbacca.md": """# Chewbacca

Chewbacca, known as Chewie, was a Wookiee warrior and co-pilot of the Millennium Falcon.
He was the loyal partner and best friend of Han Solo for over thirty years.

Wookiees are tall, fur-covered beings from the forest planet Kashyyyk.
Chewbacca was born on Kashyyyk and fought in the Clone Wars alongside Yoda when the Separatists invaded.
He was enslaved by the Galactic Empire after Order 66 and the Empire's subjugation of Kashyyyk.

Chewbacca escaped slavery and eventually met Han Solo, forming a life-debt bond with him.
He served as co-pilot of the Millennium Falcon, maintaining the ship's systems and providing formidable combat support.
Chewbacca wielded a bowcaster — a Wookiee weapon that fired explosive quarrels.

He participated in the Battle of Yavin, the Battle of Hoth, and the Battle of Endor.
After Han Solo's death at the hands of Kylo Ren, Chewbacca continued to serve the Resistance.
""",

    "r2d2.md": """# R2-D2

R2-D2 was an astromech droid who served the Galactic Republic, the Rebel Alliance, and the Resistance.
He was one of the most celebrated droids in galactic history, known for his bravery and resourcefulness.

R2-D2 was manufactured by Industrial Automaton and served aboard Queen Padme Amidala's royal starship on Naboo.
He later became the astromech unit for Anakin Skywalker's Jedi starfighter during the Clone Wars.

R2-D2 carried the stolen Death Star plans inside his memory banks, delivering them to the Rebel Alliance.
He served as Luke Skywalker's astromech during the Battle of Yavin, helping to navigate the X-Wing.
R2-D2 could interface with computer systems, project holographic messages, and perform emergency repairs.

He was one of the few beings who knew the full history of the Skywalker family.
R2-D2 and his counterpart C-3PO were inseparable companions throughout the Galactic Civil War and beyond.
""",

    "c3po.md": """# C-3PO

C-3PO was a protocol droid fluent in over six million forms of communication.
He was built by the young Anakin Skywalker on Tatooine from salvaged parts.

C-3PO served as a translator and etiquette advisor throughout the Clone Wars and the Galactic Civil War.
He was paired with the astromech droid R2-D2, and the two formed one of the most famous droid partnerships in history.

C-3PO was captured and disassembled by Jawas on Tatooine before being purchased by Owen Lars.
He served Luke Skywalker and later Princess Leia, providing translation services and protocol guidance.
During the Battle of Endor, C-3PO was mistaken for a deity by the Ewoks, which helped the Rebel Alliance
gain the Ewoks' support in the battle against the Galactic Empire.

C-3PO had his memory wiped at the end of the Clone Wars to protect the secret of Luke and Leia's birth.
He later had his memory partially restored by R2-D2 during the conflict with the First Order.
""",

    "death_star.md": """# Death Star

The Death Star was a moon-sized space station and superweapon developed by the Galactic Empire.
It was capable of destroying entire planets with a single focused superlaser blast.

The Death Star was conceived by the Separatist engineer Geonosian Poggle the Lesser and later developed by the Empire.
Its construction took decades and required enormous resources, including kyber crystals to power its superlaser.
The station was commanded by Grand Moff Tarkin and served as a symbol of Imperial power and terror.

The first Death Star was destroyed during the Battle of Yavin when Luke Skywalker fired a proton torpedo
into a small thermal exhaust port, triggering a chain reaction that destroyed the station.
The plans for the Death Star were stolen by Rebel spies on Scarif and delivered to Princess Leia.

A second Death Star was constructed above the forest moon of Endor.
It was destroyed during the Battle of Endor when Lando Calrissian piloted the Millennium Falcon
into its reactor core, triggering its destruction.
""",

    "millennium_falcon.md": """# Millennium Falcon

The Millennium Falcon was a heavily modified YT-1300 light freighter and one of the most famous ships in the galaxy.
It was captained by Han Solo and co-piloted by Chewbacca for many years.

The Millennium Falcon was originally owned by Lando Calrissian, who lost it to Han Solo in a game of sabacc.
Han Solo made numerous modifications to the ship, including an illegal hyperdrive upgrade that made it
capable of completing the Kessel Run in less than twelve parsecs.

The ship was equipped with a powerful deflector shield, concealed smuggling compartments,
and a pair of quad laser cannons. Its hyperdrive was notoriously unreliable but extremely fast when functional.

The Millennium Falcon played a crucial role in the Battle of Yavin, the Battle of Hoth, and the Battle of Endor.
It was later used by Rey and Chewbacca to fight against the First Order and the Sith Eternal fleet at Exegol.
""",

    "the_force.md": """# The Force

The Force was an energy field generated by all living things that bound the galaxy together.
It could be felt and manipulated by Force-sensitive beings, most notably the Jedi and the Sith.

The Force had two aspects: the light side, associated with peace, knowledge, and serenity;
and the dark side, associated with fear, anger, and aggression.
The Jedi Order taught its members to use the light side of the Force for knowledge and defense.
The Sith sought to harness the dark side for power and domination.

Force-sensitive beings could perform extraordinary feats: moving objects with telekinesis,
sensing the future through Force visions, communicating across vast distances,
and enhancing their physical abilities beyond normal limits.

The midi-chlorians were microscopic life forms that lived in the cells of all living beings
and communicated with the Force. A high midi-chlorian count indicated strong Force sensitivity.
Anakin Skywalker had the highest midi-chlorian count ever recorded, leading some to believe
he was the Chosen One destined to bring balance to the Force.
""",

    "jedi_order.md": """# Jedi Order

The Jedi Order was an ancient organization of Force-sensitive warriors who served as peacekeepers
for the Galactic Republic for thousands of years.

The Jedi were headquartered at the Jedi Temple on Coruscant, where younglings were trained from infancy.
They followed a strict code: "There is no emotion, there is peace. There is no ignorance, there is knowledge."
Jedi were forbidden from forming personal attachments, as attachment was seen as a path to the dark side.

The Jedi Order was governed by the Jedi High Council, which included twelve senior Jedi Masters.
Notable members included Yoda, Mace Windu, Obi-Wan Kenobi, Qui-Gon Jinn, and Anakin Skywalker.

The Order was nearly destroyed when Emperor Palpatine issued Order 66,
commanding all clone troopers to turn on and kill their Jedi commanders simultaneously.
Only a handful of Jedi survived, including Yoda and Obi-Wan Kenobi.

Luke Skywalker later attempted to rebuild the Jedi Order, but his new temple was destroyed
by his nephew Ben Solo, who had turned to the dark side and become Kylo Ren.
""",

    "sith.md": """# Sith

The Sith were an ancient order of Force-sensitive warriors who embraced the dark side of the Force.
They were the ancient enemies of the Jedi Order and sought to dominate the galaxy.

The Sith followed the Rule of Two: there could only be one Sith Master and one Sith apprentice at any time.
This rule was established by Darth Bane after the Sith nearly destroyed themselves through infighting.
The Master held the power, while the apprentice craved it — this tension was meant to ensure
only the strongest Sith survived.

Notable Sith Lords included Darth Sidious (Emperor Palpatine), Darth Vader (Anakin Skywalker),
Darth Maul, Count Dooku (Darth Tyranus), and Darth Plagueis.
Sith wielded red lightsabers, the color produced by "bleeding" kyber crystals through the dark side.

The Sith were believed to have been destroyed after the Battle of Endor,
but Emperor Palpatine secretly survived on the hidden Sith world of Exegol,
where he built the Sith Eternal fleet and planned his return to power.
""",

    "tatooine.md": """# Tatooine

Tatooine was a desert planet located in the Outer Rim Territories, orbiting twin suns.
It was one of the most remote and lawless planets in the galaxy, largely controlled by the Hutt crime syndicate.

The planet's surface was covered in vast deserts and rocky canyons, with temperatures reaching extreme highs.
Its native inhabitants included the Tusken Raiders (Sand People) and the Jawas,
small scavengers who collected and sold droids and technology.

Tatooine was the birthplace of Anakin Skywalker, who was born into slavery there.
It was also where Luke Skywalker grew up, raised by his uncle Owen and aunt Beru on a moisture farm.
Jabba the Hutt maintained his palace on Tatooine, making it a hub for criminal activity and bounty hunters.

The planet had no significant natural resources and was largely ignored by the Galactic Republic and Empire.
This obscurity made it a useful hiding place — Obi-Wan Kenobi lived in exile there for nearly twenty years,
watching over the young Luke Skywalker from a distance.
""",

    "coruscant.md": """# Coruscant

Coruscant was the capital planet of the Galactic Republic and later the Galactic Empire.
The entire surface of the planet was covered by a single massive city, making it an ecumenopolis.

Coruscant was home to the Galactic Senate, where representatives from thousands of star systems gathered.
The Jedi Temple was also located on Coruscant, serving as the headquarters of the Jedi Order.
The planet's population numbered in the trillions, making it one of the most densely populated worlds in the galaxy.

The upper levels of Coruscant were gleaming and prosperous, home to politicians, wealthy citizens, and the elite.
The lower levels descended into darkness and poverty, becoming increasingly dangerous and lawless.
The underworld of Coruscant was a haven for criminals, smugglers, and those seeking to disappear.

After the rise of the Galactic Empire, Coruscant was renamed Imperial Center.
The Jedi Temple was converted into the Imperial Palace, and the Senate was eventually dissolved by Emperor Palpatine.
""",

    "hoth.md": """# Hoth

Hoth was a remote ice planet located in the Outer Rim Territories.
It served as the location of Echo Base, the primary base of operations for the Rebel Alliance
following the Battle of Yavin.

The planet's surface was covered in snow and ice, with temperatures dropping to extreme lows at night.
Its native fauna included the wampa, a large predatory creature, and the tauntaun, a reptilian creature
used by the Rebels as a mount in the frozen terrain.

The Rebel Alliance established Echo Base on Hoth after fleeing from the Galactic Empire.
The base was discovered by an Imperial probe droid, leading to the Battle of Hoth.
The Empire deployed AT-AT walkers to assault the base, while the Rebels used snowspeeders
and tow cables to trip the massive walkers.

The Rebels were forced to evacuate Echo Base under heavy Imperial assault.
Luke Skywalker was captured by a wampa but escaped and was rescued by Han Solo,
who used the body of a dead tauntaun to keep Luke warm in the freezing night.
""",

    "dagobah.md": """# Dagobah

Dagobah was a remote swamp planet in the Outer Rim Territories.
It was the hiding place of the Jedi Master Yoda during his self-imposed exile after the fall of the Galactic Republic.

The planet was covered in dense swamps, twisted trees, and thick fog.
It teemed with life, making it strong in the Force. This abundance of life also masked Yoda's Force presence
from the Emperor and Darth Vader, allowing him to hide undetected for nearly two decades.

Luke Skywalker traveled to Dagobah after receiving a vision from the Force ghost of Obi-Wan Kenobi.
There he trained under Yoda, learning to use the Force to lift his X-Wing from the swamp
and to sense the future through Force visions.

Dagobah contained a cave strong in the dark side of the Force.
Luke entered the cave and had a vision of himself becoming Darth Vader — a warning about the dangers of the dark side.
Yoda died peacefully on Dagobah, becoming one with the Force and leaving Luke to face Darth Vader alone.
""",

    "endor.md": """# Endor

Endor, also known as the Forest Moon of Endor, was a forested moon orbiting the gas giant Endor.
It was the site of the Battle of Endor, the decisive engagement that ended the Galactic Civil War.

The moon was home to the Ewoks, small bear-like creatures who lived in villages built high in the trees.
Despite their primitive technology, the Ewoks proved to be formidable warriors
when they allied with the Rebel Alliance against the Galactic Empire.

The second Death Star was being constructed in orbit above Endor when the Rebel Alliance launched its attack.
The Rebels planned to destroy the Death Star's shield generator on the moon's surface,
allowing their fleet to attack the station directly.

The Battle of Endor resulted in the destruction of the second Death Star,
the death of Emperor Palpatine at the hands of Darth Vader,
and the redemption and death of Darth Vader himself.
The victory at Endor effectively ended the Galactic Empire's reign of terror.
""",

    "naboo.md": """# Naboo

Naboo was a peaceful planet in the Mid Rim, known for its beautiful landscapes, rolling plains, and swamp regions.
It was the homeworld of both Padme Amidala and Emperor Palpatine.

The planet was inhabited by two sentient species: the Naboo (humans) and the Gungans,
an amphibious species who lived in underwater cities beneath Naboo's lakes.
The two species had a long history of tension but eventually formed an alliance.

Naboo was governed by an elected monarchy. Padme Amidala served as Queen of Naboo before becoming a Senator.
The planet was blockaded by the Trade Federation at the instigation of Darth Sidious,
triggering the events that led to the discovery of Anakin Skywalker on Tatooine.

Naboo was also the site of the duel between Obi-Wan Kenobi, Qui-Gon Jinn, and Darth Maul.
Qui-Gon Jinn was killed by Darth Maul during this duel, and Obi-Wan avenged his master by bisecting Maul.
""",

    "mustafar.md": """# Mustafar

Mustafar was a volcanic planet in the Outer Rim Territories, covered in rivers of lava and ash.
It was the site of the climactic duel between Obi-Wan Kenobi and Anakin Skywalker.

The planet was used by the Separatists as a base of operations during the Clone Wars.
After Emperor Palpatine issued Order 66, the Separatist leaders were lured to Mustafar
and killed by Darth Vader, ending the Clone Wars.

The duel on Mustafar was one of the most significant events in galactic history.
Obi-Wan Kenobi and Anakin Skywalker — once master and apprentice — fought a desperate battle
on platforms above rivers of lava. Obi-Wan ultimately won, leaving Anakin gravely injured.
Anakin was rescued by Emperor Palpatine and transformed into the armored Darth Vader.

Darth Vader later built a castle on Mustafar, using the planet's dark side energy to commune with the dead.
The planet held deep significance for Vader as the place where Anakin Skywalker effectively died.
""",

    "clone_wars.md": """# Clone Wars

The Clone Wars was a galaxy-wide conflict fought between the Galactic Republic and the Separatists,
also known as the Confederacy of Independent Systems.

The war began with the Battle of Geonosis, where the Separatists had secretly built a massive droid army.
The Galactic Republic responded by deploying the Clone Army — an army of clones created on the ocean planet Kamino,
all cloned from the bounty hunter Jango Fett.

The Clone Wars lasted approximately three years and involved battles across hundreds of star systems.
Jedi Knights served as generals, leading clone troopers into battle against Separatist droid armies.
Notable battles included the Siege of Mandalore, the Battle of Christophsis, and the Battle of Coruscant.

The Clone Wars was secretly orchestrated by Darth Sidious, who controlled both sides of the conflict.
The war ended when Palpatine issued Order 66, a secret command embedded in the clones' genetic programming
that caused them to turn on and kill their Jedi commanders.
This allowed Palpatine to declare himself Emperor and transform the Republic into the Galactic Empire.
""",

    "order_66.md": """# Order 66

Order 66 was a secret command programmed into the genetic structure of all clone troopers
by the Sith Lord Darth Sidious. When activated, it caused the clones to turn on and kill their Jedi commanders.

The order was part of a series of contingency orders developed during the Clone Wars.
Each clone had a bio-chip implanted in their brain that, when triggered, overrode their free will
and compelled them to execute the order without question.

Order 66 was executed near the end of the Clone Wars, simultaneously across the galaxy.
Thousands of Jedi were killed by their own troops in a matter of hours.
The Jedi Temple on Coruscant was stormed by Darth Vader and the 501st Legion of clone troopers.

Only a handful of Jedi survived Order 66, including Yoda, Obi-Wan Kenobi, and a few others.
The survivors went into hiding, scattered across the galaxy.
Order 66 effectively destroyed the Jedi Order and paved the way for the rise of the Galactic Empire.
""",

    "boba_fett.md": """# Boba Fett

Boba Fett was a legendary bounty hunter and the genetic clone of the Mandalorian bounty hunter Jango Fett.
He was one of the most feared and skilled bounty hunters in the galaxy.

Boba Fett was created on Kamino as an unaltered clone of Jango Fett, raised as his son.
He witnessed his father's death at the hands of Mace Windu during the Battle of Geonosis,
which fueled his desire for revenge against the Jedi.

Boba Fett wore Mandalorian armor inherited from his father and used a variety of weapons and gadgets,
including a jetpack, wrist-mounted rockets, and a flamethrower.
He worked for both the Galactic Empire and Jabba the Hutt, tracking down targets across the galaxy.

He was hired by Darth Vader to track the Millennium Falcon, successfully capturing Han Solo on Bespin.
Boba Fett delivered Han Solo frozen in carbonite to Jabba the Hutt on Tatooine.
He was apparently killed when he fell into the Sarlacc pit during the rescue of Han Solo,
but survived and eventually became the ruler of Tatooine's criminal underworld.
""",

    "lando_calrissian.md": """# Lando Calrissian

Lando Calrissian was a smuggler, gambler, and administrator who became a general in the Rebel Alliance.
He was the former owner of the Millennium Falcon and an old friend of Han Solo.

Lando Calrissian won the Cloud City mining colony on the gas planet Bespin in a game of sabacc.
He served as Baron Administrator of Cloud City, overseeing its operations and keeping it neutral
in the conflict between the Galactic Empire and the Rebel Alliance.

When Darth Vader arrived at Cloud City, Lando was forced to betray Han Solo and his companions
to protect his city and its people. He later helped Princess Leia rescue Han Solo from Jabba the Hutt.

Lando joined the Rebel Alliance and was given the rank of General.
He piloted the Millennium Falcon during the Battle of Endor, leading the assault on the second Death Star.
He flew into the Death Star's reactor core and destroyed it, escaping just before the explosion.
""",

    "mace_windu.md": """# Mace Windu

Mace Windu was a senior member of the Jedi High Council and one of the most powerful Jedi Masters of his era.
He was known for his mastery of the Vaapad lightsaber combat form and his distinctive purple lightsaber.

Mace Windu was born on Haruun Kal and joined the Jedi Order as an infant.
He rose to become one of the most respected members of the Jedi Council,
serving alongside Yoda as one of the Order's most senior leaders.

During the Clone Wars, Mace Windu led Jedi forces in numerous battles.
He killed the bounty hunter Jango Fett during the Battle of Geonosis.
He was one of the few Jedi who was suspicious of Chancellor Palpatine's growing power.

When Anakin Skywalker revealed that Palpatine was a Sith Lord, Mace Windu led a team of Jedi Masters
to arrest him. He defeated Palpatine in combat but was betrayed by Anakin Skywalker,
who cut off his hand. Palpatine then used Force lightning to throw Mace Windu out of a window to his death.
""",

    "qui_gon_jinn.md": """# Qui-Gon Jinn

Qui-Gon Jinn was a maverick Jedi Master known for his unconventional views and his deep connection to the living Force.
He was the master of Obi-Wan Kenobi and the discoverer of Anakin Skywalker.

Qui-Gon Jinn was trained by Count Dooku before Dooku's fall to the dark side.
He was known for his willingness to challenge the Jedi Council's decisions when he believed they were wrong.
The Council respected his abilities but was wary of his independent nature.

Qui-Gon discovered Anakin Skywalker on Tatooine and immediately recognized his extraordinary Force potential.
He believed Anakin was the Chosen One destined to bring balance to the Force.
The Jedi Council initially refused to allow Anakin's training, but Qui-Gon was determined to train him.

Qui-Gon was killed by Darth Maul during the Battle of Naboo, becoming one of the first Jedi
to be killed by a Sith in a thousand years. His dying wish was for Obi-Wan to train Anakin.
Qui-Gon later discovered the secret of retaining consciousness after death,
becoming the first Jedi to communicate as a Force ghost.
""",

    "anakin_skywalker.md": """# Anakin Skywalker

Anakin Skywalker was a legendary Jedi Knight who fell to the dark side of the Force and became Darth Vader.
He was believed to be the Chosen One — a being prophesied to bring balance to the Force.

Anakin was born on Tatooine to Shmi Skywalker, with no father — a fact that led Qui-Gon Jinn to believe
he was conceived by the midi-chlorians themselves. He had the highest midi-chlorian count ever recorded.
He was a slave on Tatooine before being discovered by Qui-Gon Jinn and brought to the Jedi Order.

Anakin was trained by Obi-Wan Kenobi and became one of the most powerful Jedi of his generation.
He secretly married Senator Padme Amidala, violating the Jedi Code's prohibition on attachment.
His fear of losing Padme drove him to seek power beyond the Jedi Order's teachings.

Emperor Palpatine manipulated Anakin's fears, promising him the power to save Padme from death.
Anakin turned to the dark side, becoming Darth Vader and helping to destroy the Jedi Order.
He was ultimately redeemed by his son Luke Skywalker, destroying Emperor Palpatine at the cost of his own life.
""",

    "kylo_ren.md": """# Kylo Ren

Kylo Ren, born Ben Solo, was the son of Han Solo and Princess Leia Organa.
He was seduced to the dark side by Supreme Leader Snoke and became a commander of the First Order.

Ben Solo was the nephew of Luke Skywalker and trained at his Jedi academy.
He was seduced by the dark side and destroyed Luke's temple, killing most of his fellow students.
He took the name Kylo Ren and joined the First Order, serving as a Knight of Ren.

Kylo Ren was conflicted between the light and dark sides of the Force throughout his life.
He killed his father Han Solo in an attempt to fully commit to the dark side,
but the act haunted him. He wielded a crossguard lightsaber with an unstable red blade.

Kylo Ren was eventually redeemed through his connection with Rey and the sacrifice of his mother Leia.
He returned to the light side as Ben Solo and helped Rey defeat Emperor Palpatine on Exegol,
sacrificing his own life to resurrect Rey after she was killed by Palpatine's Force lightning.
""",

    "rey.md": """# Rey

Rey was a Force-sensitive scavenger from the desert planet Jakku who became the last Jedi
and defeated Emperor Palpatine, ending the Sith once and for all.

Rey grew up alone on Jakku, scavenging parts from crashed Star Destroyers and other wreckage
left over from the Battle of Jakku. She was abandoned there as a child by her parents,
who were hiding her from Emperor Palpatine — her grandfather.

Rey's journey began when she encountered the droid BB-8 and the former Stormtrooper Finn.
She discovered she was Force-sensitive and eventually sought out Luke Skywalker for training.
She trained briefly under Luke on the ocean planet Ahch-To and later received guidance from Leia.

Rey wielded a blue lightsaber inherited from Luke Skywalker and later constructed her own yellow lightsaber.
She had an unusual Force bond with Kylo Ren, allowing them to communicate and even transfer objects across space.
Rey defeated Emperor Palpatine on Exegol by channeling the power of all past Jedi,
ending the Sith and fulfilling the prophecy of the Chosen One.
""",

    "first_order.md": """# First Order

The First Order was a military and political organization that rose from the ashes of the Galactic Empire.
It was formed by Imperial remnants who fled to the Unknown Regions after the Battle of Endor.

The First Order was secretly guided by Supreme Leader Snoke, who was himself a puppet of Emperor Palpatine.
It built its military forces in secret, violating the Galactic Concordance that ended the Galactic Civil War.
The First Order constructed the Starkiller Base — a planet converted into a superweapon
capable of destroying entire star systems with a single shot.

The First Order destroyed the New Republic's capital and fleet with Starkiller Base,
eliminating the legitimate government of the galaxy in a single strike.
It then launched a campaign to destroy the Resistance, the military force led by General Leia Organa.

Starkiller Base was destroyed by the Resistance, and Supreme Leader Snoke was killed by Kylo Ren.
Kylo Ren then declared himself Supreme Leader of the First Order.
The First Order was ultimately defeated at the Battle of Exegol when the galaxy rose up against it.
""",

    "mandalorians.md": """# Mandalorians

The Mandalorians were a warrior culture from the planet Mandalore, known for their distinctive armor
and their code of honor. They were among the most feared warriors in the galaxy.

Mandalorian culture was built around combat, honor, and the concept of the "Mandalorian Creed."
Their iconic armor, known as Beskar armor, was made from a nearly indestructible metal called Beskar.
Mandalorians were not a single species but a culture — anyone who followed the Mandalorian way could be Mandalorian.

The Mandalorians had a complex history with the Jedi Order, having fought numerous wars against them.
During the Clone Wars, Mandalore was governed by the pacifist Duchess Satine Kryze,
but the planet was later taken over by the Death Watch, a Mandalorian warrior faction.

After the fall of the Galactic Empire, the Mandalorians were scattered across the galaxy.
Din Djarin, known as the Mandalorian, was a bounty hunter who followed the ancient Mandalorian Creed.
He became the protector of Grogu, a Force-sensitive child of Yoda's species, and eventually reclaimed
the Darksaber — the symbol of Mandalorian leadership.
""",

    "galactic_empire.md": """# Galactic Empire

The Galactic Empire was an authoritarian government that ruled the galaxy for approximately twenty years,
from the end of the Clone Wars until its defeat at the Battle of Endor.

The Empire was established by Emperor Palpatine after he declared himself Emperor,
transforming the democratic Galactic Republic into a totalitarian state.
The Senate was dissolved, and power was concentrated in the hands of the Emperor and his regional governors.

The Empire maintained control through fear and military force.
Its military included millions of Stormtroopers, thousands of Star Destroyers,
and the ultimate weapon of terror — the Death Star.
The Empire used the Death Star to destroy the planet Alderaan as a demonstration of its power.

The Empire was opposed by the Rebel Alliance, a coalition of freedom fighters
who fought to restore the Galactic Republic. The Rebel Alliance achieved a decisive victory
at the Battle of Endor, killing Emperor Palpatine and destroying the second Death Star.
Imperial remnants continued to fight for years after Endor before eventually surrendering.
""",

    "rebel_alliance.md": """# Rebel Alliance

The Rebel Alliance, formally known as the Alliance to Restore the Republic,
was a resistance movement that fought against the Galactic Empire.

The Alliance was founded by former Imperial Senator Mon Mothma and other disillusioned politicians
who opposed Emperor Palpatine's authoritarian rule. It drew members from across the galaxy —
soldiers, pilots, politicians, and ordinary citizens who believed in freedom and democracy.

The Alliance operated from hidden bases, constantly moving to avoid Imperial detection.
Its most famous bases included Echo Base on Hoth and the headquarters on the jungle moon of Yavin 4.
The Alliance's military included starfighter squadrons, ground troops, and a small fleet of capital ships.

The Alliance's greatest victory was the Battle of Yavin, where Luke Skywalker destroyed the first Death Star.
This victory gave the Alliance hope and inspired more systems to join the fight against the Empire.
The Alliance ultimately defeated the Empire at the Battle of Endor,
leading to the signing of the Galactic Concordance and the formal end of the Galactic Civil War.
""",

    "lightsaber.md": """# Lightsaber

The lightsaber was the signature weapon of the Jedi and the Sith, consisting of a plasma blade
powered by a kyber crystal and emitted from a metal hilt.

Lightsabers were constructed by Force-sensitive beings as part of their training.
The kyber crystal at the heart of each lightsaber was attuned to its wielder through the Force.
Jedi crystals naturally produced blue or green blades, while Sith "bled" their crystals through the dark side,
turning them red.

Different lightsaber colors carried different meanings:
Blue was the most common color for Jedi, associated with the Jedi Guardian role.
Green was associated with Jedi Consulars who focused on Force abilities.
Purple was extremely rare, wielded by Mace Windu, indicating a balance between light and dark.
Yellow was used by Jedi Temple Guards and Jedi Sentinels.
Red was exclusively used by the Sith, produced by corrupting a kyber crystal.

Lightsaber combat was divided into seven forms, each with different strengths and weaknesses.
The weapon could deflect blaster bolts, cut through almost any material, and was deadly in skilled hands.
""",

    "hyperspace.md": """# Hyperspace

Hyperspace was an alternate dimension that allowed starships to travel faster than light
across the vast distances of the galaxy.

Ships entered hyperspace using a hyperdrive, a specialized engine that propelled the vessel
into this parallel dimension. In hyperspace, ships could travel between star systems in hours or days
rather than the years it would take at sublight speeds.

Navigation in hyperspace was complex and dangerous. Ships had to plot precise courses
to avoid collisions with stars, planets, and other obstacles.
Astromech droids like R2-D2 were often used to calculate hyperspace jumps quickly and accurately.

The Millennium Falcon was famous for its modified hyperdrive, which made it one of the fastest ships in the galaxy.
Han Solo claimed to have made the Kessel Run in less than twelve parsecs — a measure of the shorter route
he was able to navigate through the Maelstrom, a region of dangerous black holes.

Hyperspace travel was not without risks. Ships could be pulled out of hyperspace by gravity wells,
a technology exploited by the Empire using Interdictor cruisers to trap Rebel ships.
""",

    "stormtroopers.md": """# Stormtroopers

Stormtroopers were the elite shock troops of the Galactic Empire, serving as the primary military force
of Emperor Palpatine's regime.

Originally, the Empire's military was composed of clone troopers — soldiers cloned from Jango Fett.
Over time, the Empire transitioned to recruiting ordinary humans, training them as Stormtroopers.
Stormtroopers wore distinctive white armor and carried blaster rifles as their standard weapon.

Stormtroopers were organized into legions and deployed across the galaxy to maintain Imperial control.
They were known for their intimidating appearance but were often portrayed as poor marksmen in combat.
Elite units included Death Troopers (black armor, used for classified operations)
and Scout Troopers (lighter armor, used for reconnaissance and patrol).

After the fall of the Galactic Empire, the First Order created a new generation of Stormtroopers.
Unlike Imperial Stormtroopers, First Order Stormtroopers were taken from their families as children
and conditioned from birth to serve the First Order.
FN-2187, later known as Finn, was a First Order Stormtrooper who defected to the Resistance.
""",
}

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def load_terms_map(path: Path) -> dict[str, str]:
    """Загружает и разворачивает словарь замен из JSON-файла.

    Все категории (characters, locations, factions, technology, species)
    объединяются в единый плоский словарь {original: replacement}.
    Ключ '_comment' игнорируется.
    """
    with path.open(encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    flat: dict[str, str] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def build_pattern(terms: dict[str, str]) -> re.Pattern:
    """Строит единое регулярное выражение для замены всех терминов.

    Архитектурный выбор:
    - Сортировка по убыванию длины (longest-match-first) гарантирует,
      что «Darth Vader» заменяется раньше, чем «Vader» или «Darth».
    - re.IGNORECASE позволяет ловить «DARTH VADER», «darth vader» и т.д.
    - re.escape защищает спецсимволы (например, «R2-D2», «C-3PO»).
    """
    sorted_keys = sorted(terms.keys(), key=len, reverse=True)
    pattern_str = "|".join(re.escape(k) for k in sorted_keys)
    return re.compile(pattern_str, flags=re.IGNORECASE)


def make_replacer(terms: dict[str, str]):
    """Возвращает callback для re.sub, сохраняющий регистр первой буквы."""
    lower_map = {k.lower(): v for k, v in terms.items()}

    def replacer(match: re.Match) -> str:
        original = match.group(0)
        replacement = lower_map.get(original.lower(), original)
        # Сохраняем регистр первой буквы оригинала
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement[0].lower() + replacement[1:]

    return replacer


def obfuscate_text(text: str, pattern: re.Pattern, replacer) -> str:
    """Применяет обфускацию ко всему тексту."""
    return pattern.sub(replacer, text)


def process_articles(
    articles: dict[str, str],
    pattern: re.Pattern,
    replacer,
    output_dir: Path,
) -> list[str]:
    """Обфусцирует все статьи и сохраняет их в output_dir.

    Returns:
        Список имён созданных файлов.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    for filename, raw_text in articles.items():
        obfuscated = obfuscate_text(raw_text, pattern, replacer)
        out_path = output_dir / filename
        out_path.write_text(obfuscated, encoding="utf-8")
        created.append(filename)
        print(f"  ✓ {filename}")

    return created


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    """Основная функция: загружает словарь, строит паттерн, обфусцирует статьи."""
    print("=" * 60)
    print("QuantumForge RAG — Обфускация базы знаний")
    print("=" * 60)

    # 1. Загружаем словарь замен
    print(f"\n[1/3] Загрузка словаря замен из {TERMS_MAP_PATH} ...")
    terms = load_terms_map(TERMS_MAP_PATH)
    print(f"      Загружено {len(terms)} терминов для замены.")

    # 2. Компилируем паттерн
    print("\n[2/3] Компиляция регулярного выражения (longest-match-first) ...")
    pattern = build_pattern(terms)
    replacer = make_replacer(terms)
    print(f"      Паттерн содержит {len(terms)} альтернатив.")

    # 3. Обфусцируем и сохраняем
    print(f"\n[3/3] Обфускация {len(RAW_ARTICLES)} статей → {OUTPUT_DIR}")
    created = process_articles(RAW_ARTICLES, pattern, replacer, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"Готово! Создано {len(created)} файлов в папке '{OUTPUT_DIR.name}/'.")
    print("=" * 60)


if __name__ == "__main__":
    main()
