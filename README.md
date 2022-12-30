# 📈 Kontrolling (PEVS-PANI)
⚙️ Zdrojové kódy a projekty z predmetu Kontrolling na DNN (Deep Neural Network) na PEVS/PANI

## Trénovacie a Validačné Vzorky
Na vstupe (vstupnej vrstve) máme RGB obrázky o veľkosti 128 x 128 px, 3 kanály. Veľkosť trénovacích vzoriek je 1098 a počet validačných vzoriek je 277. Rozdelenie je teda 80 % trénovacie vzorky a 20 % validačné t.j. paretovo pravidlo 80/20.

|     Celkový počet   vzoriek:         |     1370    |     100%    |
|--------------------------------------|-------------|-------------|
|     Počet trénovacích   vzoriek:     |     1098    |     80%     |
|     Počet validačných   vzoriek:     |     272     |     20%     |

## Použité experimenty
### Experiment 000
```python
{
   # Nazov experimentu
   "name": "exp_000",
   # Počet konvolučných blokov
   "conv_num": 2,
   # V rámci konvolučných vrstviev, používame rôzne typy filtrov
   "filter_num": 32,
   # Rôzne velkosti filtrov
   "filter_size": (3, 3),
   # Veľkosť max_poolingu, aby sme redukovali výsledné matematické operácie, ktoré prebehli cez konvolúcie
   # a potom výsledné matice, akým spôsobom sa zgrupovali a zmenšovali, aby sa neuronová sieť rýchlejšie natrénovala
   # a lepšie rozpoznávala jednotlivé oblasti
   "max_pooling": (2, 2),
   # Regularizačná technika, pri prepojených/plne prepojených vrstvách bude náhodne dávať váhu
   # jednotlivých prepojení na 0, čiže ako keby ich vypol
   # Regularizačný prvok, vďaka týmto výpadkom je neuronová sieť vyslovene nútená hladať prepojenia vo vzoroch
   # Môže tam byť nejaký šum
   # 0.2 znamená, že 20 % prepojení nám náhodne vypadne, dropoutom sa vynulujú tie prepojenie, nevymažú sa vyslovene
   "dropout": 0.2,
   # Optimalizátor, najzákladnejší
   "optimizer": "adam",
   # Veľkosť dávky, v ktorom neuronová sieť bude jednotlivé informácie vyhodnocovať
   # na koľko obrázkov sa naraz pozrie než sa aktualizuje
   "batch_size": 16,
   # Ak je True využite sa funkcia maxpooling, ak je False využije sa averagepooling
   "is_max_pooling": True,
}
```
![exp_000](https://user-images.githubusercontent.com/24510943/210070243-b4430452-2064-48ba-a90c-84dbdd455262.png)

### Experiment 001 a 002
```
python
# Rozdiel medzi exp_000, exp_001 a exp_002:
# - Rôzne velkosti filtrov (filter_size) 3,3 a 5,5
# - Rôzne is_max_pooling exp_000 použije maxpooling exp_001 použije averagepooling
{
    "name": "exp_001",
    "conv_num": 2,
    "filter_num": 32,
    "filter_size": (5, 5),
    "max_pooling": (2, 2),
    "dropout": 0.2,
    "optimizer": "adam",
    "batch_size": 16,
    "is_max_pooling": False,
},
{
    "name": "exp_002",
    "conv_num": 2,
    "filter_num": 32,
    "filter_size": (3, 3),
    "max_pooling": (2, 2),
    "dropout": 0.2,
    "optimizer": "adam",
    "batch_size": 16,
    "is_max_pooling": False,
}
```
![exp_001a002](https://user-images.githubusercontent.com/24510943/210070598-00283f24-f3b5-41a6-abd5-f840efa56b0e.png)


## 🐞 Problémy (bugy), na ktoré som narazil

1. Problém s najnovšou verziou PyCharm 2022.3 s Jetbrains ToolBox (aj Community aj Professional edícia) s portami pri nainštalovaní Hyper-V, Docker   

**Riešenie:**  
winrat a nepoužívať Jetbrains Toolbox    

```
net stop winnat
net start winnat
```

2. Problém s nezobrazovaním balíkov v Python Interpreteri, prázdne okno dostupné balíčky (Available Packages)
![Python-Interpreter-baliky](https://user-images.githubusercontent.com/24510943/210062678-c91a4595-2b0a-45ea-93e2-c705c769a453.png)

![Python-Interpreter-baliky-NG](https://user-images.githubusercontent.com/24510943/210062685-535f2e93-3a6a-43e2-9e74-9df6d9ced886.png)

3. Problém s verziu Pythonu v interpreteri  

**Riešenie:**    
Treba použiť v Interpreteri pre Conda Python 3.8 t.j. downgrade verzie z 3.9 až 3.10.  

4. Problém s duplicitou knižnic
https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a

```
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5 already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. 
# That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.

```
**Riešenie:**  
```python
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  
```

alebo vymazať libiomp5md.dll  

5. Problém s vygenerovaním dát pre 1. epochu (Invalid Argument Error / Graph Execution Error)
https://stackoverflow.com/questions/71153492/invalid-argument-error-graph-execution-error

**Riešenie:**   
Dense pri vrstvách z Kerasu nastaviť na rovnaký počet ako má byť priečinkov/tried. Pri psoch a mačkach boli logicky 2, pri 10 druhov opíc na 10.  

6. Problém s nesprávnou veľkosťou validačnej zložky miesto 272 bola 2720...  
**Riešenie:**    
Jednoduchá kontrola či máme pomer 80/20 alebo 70/30 (Trenovácie/Validačné) alebo (Trenovácie/Validačné+Testovacie)   

## 📚 Dôležité zdroje/dokumentácia  
1. [TensorFlow Keras dokumentácia](https://www.tensorflow.org/api_docs/python/tf/keras)  
2. [Keras Optimalizátory](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)  
3. [Keras Aktivačné Funkcie](https://www.tensorflow.org/api_docs/python/tf/keras/activations)  
4. [Dataset Opice 10 Monkey Species z Kaggle](https://www.kaggle.com/datasets/slothkong/10-monkey-species)  
5. [Ahmed Gaber Convolutional neural network z Kaggle](https://www.kaggle.com/code/gaber0512/monkey-species-convolutional-neural-network)  
6. [Képešiová Zuzana - Inteligentné metódy diagnostiky a riadenia mechatronických systémov](https://www.fei.stuba.sk/buxus/docs/2020/Kepesiova_autoreferat.pdf) 
