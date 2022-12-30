# Kontrolling (PEVS-PANI)
⚙️ Zdrojové kódy a projekty z predmetu Kontrolling na DNN na PEVS/PANI

## 🐞 Problémy (bugy), na ktoré som narazil:

1. Problém s najnovšou verziu PyCharm (aj Community aj Professional edícia) s portami pri nainštalovaní Hyper-V, Docker 
Riešenie: winrat

2. Problém s nezobrazovaním balíkov v Python Interpreteri, prázdne okno dostupné balíčky (Available Packages)

3. Problém s vygenerovaním dát pre 1. epochu (Invalid Argument Error / Graph Execution Error)
https://stackoverflow.com/questions/71153492/invalid-argument-error-graph-execution-error

Riešenie:
Dense pri vrstvách z Kerasu nastaviť na rovnaký počet ako má byť priečinkov/tried. Pri psoch a mačkach boli logicky 2, pri 10 druhov opíc na 10.

