# **🐦 Flappy Bird AI** 

Ez a projekt egy **Flappy Bird klón**, amely megtanul játszani önmagától a **Genetikus Algoritmus (Genetic Algorithm)** és **Neurális Hálók (Neural Networks)** segítségével. A projekt tiszta Python nyelven íródott pygame használatával, külső gépi tanulási könyvtárak (pl. TensorFlow, PyTorch) nélkül. 

A mesterséges intelligencia generációról generációra fejlődik, a természetes kiválasztódás elveit utánozva, amíg el nem éri a tökéletes játékmenetet.

## **🎥 Demó**

*(Hamarosan: Egy GIF animáció a működő AI-ról)*

## **📋 Tartalomjegyzék**

* [Hogyan működik? \- Az Elmélet]()
  * [A Neurális Háló]()  
  * [Bemenetek (Szenzorok)]()  
  * [A Genetikus Algoritmus]()  
* [Telepítés és Futtatás]()  
* [Fájlok szerkezete]()

## **🧠 Hogyan működik? \- Az Elmélet**

A játék minden egyes madara rendelkezik egy saját "aggyal" (neurális hálóval). Ez a hálózat dönti el minden egyes képkockában, hogy a madárnak **ugrania kell-e vagy sem**.

### **A Neurális Háló**

A hálózat topológiája **4-3-1**, ami azt jelenti, hogy:

* **Bemeneti réteg (Input):** 4 neuron (a madár érzékszervei).  
* **Rejtett réteg (Hidden):** 3 neuron (a döntéshozatal komplexitása, ReLU aktivációval).  
* **Kimeneti réteg (Output):** 1 neuron (döntés: ugrás vagy sem, Sigmoid aktivációval).

Ha a kimeneti érték \> 0.7, a madár ugrik.

### **Bemenetek (Szenzorok)**

A madár a következő 4 adatot látja a világból (az értékek normalizálva vannak a jobb tanulás érdekében):

1. **Madár Y pozíciója:** Milyen magasan van a madár (0-1 skálán).  
2. **Madár sebessége:** Milyen gyorsan zuhan vagy emelkedik.  
3. **Távolság a csőtől (X):** Milyen messze van a következő akadály vízszintesen.  
4. **Függőleges távolság a nyílástól (Y):** Hol van a madár a cső nyílásának közepéhez képest.

### **A Genetikus Algoritmus**

A tanulás folyamata a biológiai evolúciót utánozza:

1. **Populáció létrehozása:** Kezdetben 50 "buta" madarat hozunk létre véletlenszerű agyi kapcsolatokkal (súlyokkal).  
2. **Szelekció (Fitness):** A madarak játszanak. Aki tovább él és több csövön jut át, magasabb "fitness" pontszámot kap.  
   * \+50 pont minden sikeres csőért.  
   * \+0.1 pont minden túlélt képkockáért.  
   * Büntetés a felesleges ugrálásért.  
3. **Kiválasztás (Elitizmus):** A generáció végén a legjobban teljesítő 10 madarat (az "eliteket") változatlanul átvisszük a következő generációba.  
4. **Keresztezés (Crossover):** A maradék helyeket az előző generáció legjobbjainak "gyermekeivel" töltjük fel. A gyerek örökli a szülők súlyainak keverékét.  
5. **Mutáció:** 10% eséllyel véletlenszerűen módosítjuk a súlyokat, hogy új viselkedésformákat vezessünk be (pl. "talán jobb lenne kicsit korábban ugrani").

## **🚀 Telepítés és Futtatás**

### **Előfeltételek**

Szükséged lesz a Python telepítésére (3.x verzió).

### **1\. Klónozás vagy letöltés**

Töltsd le a kódot vagy klónozd a repót.

### **2\. Függőségek telepítése**

A projekthez csak a pygame könyvtár szükséges. Telepítsd a requirements.txt segítségével:

pip install \-r requirements.txt

### **3\. A játék indítása**

Futtasd a Python fájlt:

python flappyAI.py

**Megjegyzés a grafikáról:** A programhoz tartozik egy images mappa a játék grafikáival. Ha a mappa vagy a benne lévő fájlok hiányoznak, a kód automatikusan létrehoz egy mappát és helyettesítő színes négyzeteket generál, így a játék grafika nélkül is futtatható. A legjobb élmény érdekében azonban érdemes használni a mellékelt képeket.

## **📂 Fájlok szerkezete**

* flappyAI.py: A fő programkód, amely tartalmazza a játék logikáját, a neurális hálót és a genetikus algoritmust.  
* requirements.txt: A szükséges Python csomagok listája.  
* images/: A játék grafikáit tartalmazó mappa (pl. bird.png, pipe.png, background.png, ground.png).  
* FORRASOK.md: A projekthez használt források és inspirációk listája.

*Készítette: Rácz László \- CI880V*