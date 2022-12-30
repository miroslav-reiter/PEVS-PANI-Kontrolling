# Kontrolling (PEVS-PANI)
⚙️ Zdrojové kódy a projekty z predmetu Kontrolling na DNN na PEVS/PANI

## 🐞 Problémy (bugy), na ktoré som narazil:

1. Problém s najnovšou verziu PyCharm (aj Community aj Professional edícia) s portami pri nainštalovaní Hyper-V, Docker 
Riešenie: winrat

```
net stop winnat
net start winnat
```

2. Problém s nezobrazovaním balíkov v Python Interpreteri, prázdne okno dostupné balíčky (Available Packages)
![Python-Interpreter-baliky](https://user-images.githubusercontent.com/24510943/210062678-c91a4595-2b0a-45ea-93e2-c705c769a453.png)

![Python-Interpreter-baliky-NG](https://user-images.githubusercontent.com/24510943/210062685-535f2e93-3a6a-43e2-9e74-9df6d9ced886.png)

3. Problém s vygenerovaním dát pre 1. epochu (Invalid Argument Error / Graph Execution Error)
https://stackoverflow.com/questions/71153492/invalid-argument-error-graph-execution-error

Riešenie:
Dense pri vrstvách z Kerasu nastaviť na rovnaký počet ako má byť priečinkov/tried. Pri psoch a mačkach boli logicky 2, pri 10 druhov opíc na 10.

