# Bitcoin-trend-elorejelzes

## Orderbook_tester.py
Ebben a scriptben tesztelem az adatfeldolgozás funkciot, az orderbook-ok keppe alakitasaval, tovabb ellenorizhetem, hogy a halozatom ertelmes adatot kap meg vagy csak zajt.

A script kiemenete a kepek mappaba, de google drivra is felraktam, mert ezeket érdemes egymás után 'végigpörgetve' megnézni.
link: https://drive.google.com/open?id=1lSkycFYcWqkTA1qWtskb3D53PZEQVlX3

Magyarázat: <br />
A kép egy sora egy orderbook. Így a képen megtekinthető az orderbook időbeli változása. <br />
Feketén kis érték, fehéren nagy értékek szerepelnek. <br />
Így a középső fekete csík jobb oldalan találhatóak a cumulative asks-ok és a bal oldalon pedig a cumulative bids-ek.<br />
A képek továbbá jól mutatják az orderbook-ok asszimetráját tehát, hogy asks-ból vagy bids-ből van több. 


## Orderbook.py
Tartalmazza a konvolúciós hálót és feldolgozza az orederbook-okat.  <br />
Háló: konvolúciós háló,
X - Orderbook-ok, egyszerre 40 Orderbook-ot kap meg a háló
y - trend, 1 ha a következő 100 adat átlaga nagyobb mint az előző 100 adat átllaga
adat itt a btc árára vonatkozik

## price.py

LSTM hálózat, a fentebb említett elv alapján meghatározott trender jósolja meg
X - btc ár adatok, egyszerre megkap ~ előző 10 at
y - trend


## Download.py

A script ami letölti & feldolgozza, majd csv formátumba elmenti az orderbook-okat és btc price adatokat
