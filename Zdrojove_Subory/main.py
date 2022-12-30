# Import modulov a balíčkov
# Modul pre tvorbu zložiek/súborov
import os

# Knižnica pre umelé neuronové siete (Súčasť TensorFlow), rozhranie pre TensorFlow
# Generátor obrázkov (už deprecated)
from keras.preprocessing.image import ImageDataGenerator
# Vstupná vrstva
from keras import Input
# Trieda model
from keras.models import Model
# Klasifikácia obrazových dát pomocou konvolučných vrstiev (2d konvolúcia)
from keras.layers import Conv2D, MaxPooling2D, AvgPool2D
# Aktivačné funkcie, Dropout pre filtrovanie, flatten vrstiev, ktoré skončia po konvolúcií
# Dense pre plneprepojenu vrstvu
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
# Spätné volania, chceme vidieť ako sa naše neuronové siete trénujú
from keras import callbacks

# Fix duplicitných knižníc alebo vymazať libiomp5md.dll
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5 already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.

# Definicia rozmerov vstupneho obrazka pre neuronovu siet vyska x sirka v px
# Mensie rozmery logicky rýchlejsí tréning
# Cim je väčší vstupný obrázok, tým je pomalší tréning aj samotná predikcia
# Neuronová sieť musí urobiť väčší počet matematických operácií
img_width, img_height = 128, 128

# Parametre pre projekt
# Cesty, kde sa nachadzajú trenovacie, validačné dáta a
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
# Logy (ukladanie modelov, trénovacie krivky, metrika úspešnost tohto modelu, spôsob akým sa mení chybová funkcia)
log_dir_root = 'logs'
# Velkosť trénovacích vzoriek, celkovo je všetkých vzoriek 1370
nb_train_samples = 1098
# Velkosť validačných vzoriek
nb_validation_samples = 272
# Počet epoch, na ktorých sa bude tento model trénovať
# Koľkokrát prebehne tým datasetom, koľkokrát sa budú aktualizovať váhy
epochs = 50

# create logdir if it does not exist
# Vytvoríme zložky Logy, do ktorej budeme ukladať tieto priebežné metriky a výsledky z trénovania
# Príznak/flag exist_ok=True je na prípad opakovaného púštania skriptu,
# ak zložka Logy existuje, tak tento krok preskočí
os.makedirs(log_dir_root, exist_ok=True)

# Zoznam Volieb, ktoré definujú jednotlivé experimenty (Slovníky)
# Nastavenia experimentov, parametre modelu a ako sa má správať

experimenty = [
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
    },
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
    },
]

for experiment in experimenty:
    # Vytvori podzložku pre Logs a to konkrétneho experimentu
    # V zložke Logs bude podzložka s názvom expirimentu
    log_dir = os.path.join(log_dir_root, experiment["name"])
    os.makedirs(log_dir, exist_ok=True)

    # Definicia modelu
    # Vstupná vrstva, ktorá má rozmery sirka, vyska a pocet kanálov 3 - RGB farebné obrázky, trojrozmerné
    # Inputs ešte ďalej používame
    inputs = Input((img_width, img_height, 3))

    ### Konvolučný blok
    # Zoberieme konvolučnú vrstvu, konvolvujeme tento vstupný obraz s rôznym počtom filtrov, ktoré sme si nastavili v parametroch (3, 3)
    # čiže velkosť okna bude 3x3, použijeme aktivačnú funkciu relu, iniciálizátor kernelu, padding same - veľkosť obrázku zostane rovnaká
    x = Conv2D(experiment["filter_num"], experiment["filter_size"], activation="relu", kernel_initializer='he_uniform', padding="same")(inputs)
    # Normalizácia, len čo prebehne konvolučná vrstva
    x = BatchNormalization()(x)
    # Rozhodovanie či sme v parametroch vybral MaxPooling alebo average pooling
    if experiment["is_max_pooling"]:
        x = MaxPooling2D(experiment["max_pooling"])(x)
    else:
        x = AvgPool2D(experiment["max_pooling"])(x)
    # Úprava váh, aby sme nepretrénovali model
    x = Dropout(experiment["dropout"])(x)
    ### END Konvolučný blok

    # Podľa ďalšieho počtu konvolučných blokov, ktoré sa nahádzajú v našej sieti, podľa toho ich tam budeme postupne pridávať
    # Počet filtrov sa potom bude postupne zväčšovať vid. * (2 ** i)
    # Keďže aktuálne máme nastavené 2, take ešte 1x zbehne tento for cyklus aby sa nastackovali ďalšie vrstvy
    # Ak bude nastavení 3, tak ešte 2x
    for i in range(1, experiment["conv_num"]):
        x = Conv2D(experiment["filter_num"] * (2 ** i), experiment["filter_size"], activation="relu", kernel_initializer='he_uniform', padding="same")(x)
        x = BatchNormalization()(x)
        if experiment["is_max_pooling"]:
            x = MaxPooling2D(experiment["max_pooling"])(x)
        else:
            x = AvgPool2D(experiment["max_pooling"])(x)
        x = Dropout(experiment["dropout"])(x)

    # Keď bude tento model všetky konvolučné bloky zahrňať, tak potom dáme Flatten vrstvu, ktorá
    # všetky naše niekoľko rozmerné vstupy, ktoré vídu z poslednej dropout vrstvy flattne na 1 vektor
    x = Flatten()(x)
    # Tento vektor plne prepoji s 128 uzalmi/nodmi a opat použije aktivačnú funkciu relu, iniciálizátor kernelu
    x = Dense(128, activation="relu", kernel_initializer='he_uniform')(x)
    x = Dropout(experiment["dropout"])(x)
    # Výsledná vrstva bude opäť plne prepojená a to bude mať 10 výstupy (lebo máme 10 druhov opíc)
    # Nie binárny problém ale kategorický problém
    x = Dense(10)(x)
    # Na výstupnú vrstvu ešte použijeme aktivačnú funkciu sigmoid
    # Budeme rozoznávať sigmoid a softmax - toto si môžeme otestovať
    outputs = Activation('sigmoid')(x)

    # Aby sme mali zadefinovaný model, dáme mu koľko má zadefinovaný vstupných vrstiev (teraz 1 vstupnú a 1 výstupnú vrstvu)
    model = Model(inputs=[inputs], outputs=[outputs])
    # Kompilácia modelu, použijeme chybovú funkciu categorical_crossentropy
    # Použiujeme náš predvolený optimizer Adam
    # Budeme sledovať accuracy/presnosť
    # Máme vyvážený dataset, kedy každý druh opic má v priemere 10 % zastúpenie
    # Ak použijete nevyvážené rozdelenie, využite MCC metriku na miesto accuracy. - nepoužívame MCC metriku ale accuracy
    model.compile(
        loss='categorical_crossentropy',
        optimizer=experiment["optimizer"],
        metrics=['accuracy'],
    )

    # print model information
    # Na záver skript vypíše sumarizáciu ako model vyzerá
    model.summary()


    # this is the augmentation configuration we will use for training
    # Ak máme zadefinovaný model, môžeme si zadefinovať generátory obrázkov
    # Máme síce datasety alebo potrebujeme dátový generátor obrázkov, ktorý bude do našej neuronovej siete
    # postupne dávkovať jednotlivé obrázky
    # už k dispozícií v rámci Tensorflow
    # Správnym nastavením parametrov, chceme predísť pretrénovaniu (neuronová sieť sa vtedy naučí riešiť
    # problém len pre trénovaciu množinu, len na namemoruje ako táto trénovacia množina vyzerá)
    # Vždy ked sa vyberie obrázok z trénovacej množiny, môže Obrázkový generátor tento obrázok upraviť/augumentovať
    # Nechceme, aby si neuronová sieť nezapamatala úplne presne tieto obrázky/aké sú tam obrázky, aby videla rovnaké obrázky
    # Generátor bude obrázky upravovať
    # Môžu sa aplikovať všetky parametre a vznkne niečo iné, neuronová sieť je tak nútená sa učiť
    train_datagen = ImageDataGenerator(
        # Normalizácia, chceme hodnoty px na 0-1 (RGB 0-255)
        rescale=1. / 255,
        # Skosenie obrázku s 20 % šancou sa nám obrázok skosí alebo sa priblíži
        shear_range=0.2,
        # Priblíženie obrázku s 20 % šancou sa nám obrázok priblíži
        zoom_range=0.2,
        # Natiahnutie obrázku do šírky s 10 % šancou sa nám obrázok natiahne do šírky
        width_shift_range=0.1,
        # Natiahnutie obrázku do výšky s 10 % šancou sa nám obrázok natiahne do výšky
        height_shift_range=0.1,
        # Môže sa obrázok horizontálne otočiť/zrkladlovo
        horizontal_flip=True,
    )

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    # Zvlášť generátor na testovanie/testovacie obrázky v našom prípade validačné
    # Nebudeme tu používať ďalšie úpravy ako zoom a natahovanie
    # Chceme zistiť nakoľko je úspešný v obrázkoch, ktoré nevidel a v obrázkoch, ktoré sú relevantné
    # Rescale lebo neuronová sieť bude očakávať hodnoty 0 až 1 a nie 0 až 255
    # Vstupy sú 3-rozmerné dáta
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # loading a generator of train inputs for neural network
    # Generátory, ktorú bude naťahovať jednotlivé obrázky z našich zložiek
    train_generator = train_datagen.flow_from_directory(
        # Pre trenovaci generator - trenovacia zlozka (directory)
        train_data_dir,
        # Rozmer v akých sa tam má obrázok nahádzať
        target_size=(img_width, img_height),
        # Velkosť dávky, dali sme 16, 16 obrázkov naraz dostane naraz neuronová sieť
        batch_size=experiment["batch_size"],
        # Budeme mať kategórie obrázkov
        class_mode='categorical',
    )

    # loading a generator of validation inputs for neural network
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),

        batch_size=experiment["batch_size"],
        class_mode='categorical',
    )

    # callbacks definition
    # Zadefinujeme si Listenery 2 objekty/callbacky
    # Vždy keď sa nátrenuje 1 epocha v prípade TensorBoardu, tak vtedy sa uložia informácie ako:
    # kód chybovej funkcie a výsledok našej sledovanej metriky (accuracy...) - toto 1 callback vždy na konci epochy
    #
    callback_list = list()

    # tensorboard callback
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    callback_list.append(tb_cb)

    # saving model at best results
    # Ďalší callback vďaka nemu získame natrénované modely
    # Bude sú aj reálne ukladať do Logovacej zložky už pre konkrétny experiment
    # Náš model sa bude volať best typu hdf5
    mcp_save = callbacks.ModelCheckpoint(
        os.path.join(log_dir, 'best.hdf5'),
        # Ukladáme len ten najlepší model, šetríme pamäť
        save_best_only=True,
        # Ako to vyhodnocuje tento callback, že toto je práve ten najlepší model, bude validačná chybová funkcia val_loss
        # Validačná chybová funkcia čím je menšia, čím je menšia chyba, tak tým je lepší model
        # Budeme to vyhodnocovať na základe validačnej vzorky
        # Čím najpresnejšie bude natrénovaný model vzhľadom k validačnej vzorke, tým budeme vedieť zadefinovať ten najlepší model
        monitor='val_loss',
        # Chceme čo najnižšiu hodnotu z validačnej chybovej funkcie
        mode='min',
    )
    # Keď budeme mať zoznam takýchto callbackov, tak môžeme prístupiť k trénovaniu modelu
    callback_list.append(mcp_save)

    # train the model
    # Model máme zadefinovaný a teraz zavoláme nad ním funkciu fit_generator a jeho parametre bude nasledovne
    model.fit_generator(
        # Generator všetkých testovacích obrázkov, ktorý bude postupne dávkovať zo zložky
        train_generator,
        # Počet krokov na epochu, podľa počtu vstupných obrázkov // velkosť dávky
        steps_per_epoch=nb_train_samples // experiment["batch_size"],
        # Počet epoch/opakovaní, ktoré má vykonať
        epochs=epochs,
        # To isté aj pre validačné vzorky
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // experiment["batch_size"],
        callbacks=callback_list,
    )

## Vzory výstupov a ich interpretácia z datasetu psy a mačky (dogs and cats)
## prikaz model.summary() vdaka, ktoremu vidíme celú štruktúru nášho modelu, ktorý sme si vytvorili
# Vyplulo to zoznam vrstiev, v akom poradí sú povytvárané a aká je ich vstupná vrstva, je tam typ vrstvy, nie presne čo dostávajú na vstupe
# Názov a typ vrstvy, potom je tu Shape/Rozmer výstupnej matice/dáta, dáva aj počet parametrov
# Output      Shape posledná hodnota 3, 32, 32, ... velkosť filtra
# Output      Shape prvá hodnota None, None, None, ... veLkosť toho batchu/dávky môže tam byť 16, 200, 32, ale keď je tam None môže to byť flexibilné

# Model: "model"
# _________________________________________________________________
# Layer(type)                 Output      Shape           Param
# =================================================================
# input_1(InputLayer)         [(None, 200, 200, 3)]       0
#
# conv2d(Conv2D)              (None, 200, 200, 32)        896
#
# batch_normalization(BatchN(None, 200, 200, 32)          128
# ormalization)
#
# max_pooling2d(MaxPooling2D(None, 100, 100, 32)          0
# )
#
# dropout(Dropout)        (None, 100, 100, 32)            0
#
# conv2d_1(Conv2D)        (None, 100, 100, 64)            18496
#
# batch_normalization_1(Batc  (None, 100, 100, 64)        256
# hNormalization)
#
# max_pooling2d_1(MaxPooling  (None, 50, 50, 64)           0
# 2D)
#
# dropout_1(Dropout)      (None, 50, 50, 64)              0
#
# flatten(Flatten)        (None, 160000)                  0
#
# dense(Dense)(None, 128)
# 20480128
# ...


## Vo všeobecnosti tam máme 20 milionov parametrov, ktoré sa tam trénujú
# =================================================================
# Total params: 20,500,162
# Trainable params: 20,499,970
# Non-trainable params: 192

## Naše generátory našli 2100 obrázkov, ktoré prislúchajú 2 triedami
# Found 2100 images belonging to 2 classes.
# Found 700 images belonging to 2 classes.

## Samotný tréning, náš model sa postupne trénuje naprieč jednotlivými epochami, zadali sme si na začiatku 50 epoch
# 131/131 sú tzv. kroky (stepy) tie sme si vypočítali koľko ich cca bude
#  92s 71s 74s koľko nám to trvalo, 92 sekúnd na 1 epochu na tú iniciačnú epochu
# loss: 6.7448 hodnotu loss funkcie a accuracy: 0.5465 na začiatku pre trénovací set , a máme val_loss: 18.0068 - val_accuracy: 0.5494 pre validačný set
# Hodnota loss funkcie by mala postupne klesať, nemusí ale vždy klesať, môže aj poskočiť t.j. vystúpiť a potom padnúť
# Celkový trend pre loss funkciu má byť klesajúci

# To isté platí aj pre accuracy (môže aj poskočiť t.j. vystúpiť a potom padnúť) ale táto bude rásť, postupne by sa mala zvyšovať a sem tam potom poskočiť/klesať
# Celkový trend pre accuracy má byť rastúci
# Hodnoty accuracy sledujeme ako krivku v Tensorborde

# 50 epoch cca 50 minút až hodinky, kým sa natrénuje 1 model
# Epoch 1/50
# 131/131 [==============================] - 92s 695ms/step - loss: 6.7448 - accuracy: 0.5465 - val_loss: 18.0068 - val_accuracy: 0.5494
# Epoch 2/50
# 131/131 [==============================] - 71s 542ms/step - loss: 0.6641 - accuracy: 0.5744 - val_loss: 23.2886 - val_accuracy: 0.5683
# Epoch 3/50
# 131/131 [==============================] - 74s 564ms/step - loss: 0.6628 - accuracy: 0.5768 - val_loss: 16.7334 - val_accuracy: 0.5756