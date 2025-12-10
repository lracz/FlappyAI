# **📚 Források és Referenciák**

Ez a projekt az alábbi források, elméletek és inspirációk alapján készült. A cél egy saját "Deep Learning" szerű mechanizmus implementálása volt külső ML könyvtárak nélkül.

## **1\. Elsődleges Inspiráció és Tutorial**

A projekt alapötlete és a megvalósítás logikája az alábbi YouTube oktatósorozat alapján készült:

* **Python Flappy Bird AI Tutorial (with NEAT)** \- Készítette: *Tech With Tim*  
  **Videó sorozat linkje:** [Megtekintés YouTube-on](https://www.youtube.com/watch?v=NPbHUyVDYDw&list=PLzMcBGfZo4-lwGZWXz5Qgta_YNX3_vLS2&index=9)  
  *Leírás: Ez a videósorozat mutatja be, hogyan lehet a NEAT (NeuroEvolution of Augmenting Topologies) elvet alkalmazva megtanítani a gépet Flappy Bird-del játszani. A projekt ezt az elméleti hátteret alkalmazza saját neurális háló implementációval.*

## **2\. Felhasznált Technológiák**

* **Pygame** Hivatalos dokumentáció: [https://www.pygame.org/docs/](https://www.pygame.org/docs/)  
  *Felhasználás: A játékmotor, grafikai megjelenítés és eseménykezelés (billentyűzet, ablak).*

## **3\. Elméleti Háttér**

* **Genetikus Algoritmusok (Genetic Algorithms)** Általános leírás: [Wikipedia \- Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)  
  *Implementáció a kódban: GeneticAlgorithm osztály (szelekció, crossover, mutáció).*  
* **Neurális Hálók (Neural Networks)** Feedforward hálózatok működése: [3Blue1Brown \- Neural Networks (YouTube)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  
  *Implementáció a kódban: NeuralNetwork osztály (súlyok mátrixa, bias, ReLU és Sigmoid aktivációs függvények).*  
* **Neuroevolúció (NEAT alapelvek)** Bár nem a teljes NEAT (NeuroEvolution of Augmenting Topologies) algoritmust használjuk, az alapötlet – a neurális hálók súlyainak evolúciós úton történő tanítása – innen származik.