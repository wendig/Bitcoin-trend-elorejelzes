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
Háló: konvolúciós háló,</br>
X - Orderbook-ok, egyszerre 40 Orderbook-ot kap meg a háló
y - trend, 1 ha a következő 100 adat átlaga nagyobb mint az előző 100 adat átlaga
(adat itt a btc árára vonatkozik)

## price.py

LSTM hálózat, a fentebb említett elv alapján meghatározott trendet jósolja meg<br />
X - btc ár adatok, egyszerre megkap ~ előző 10 at</br>
y - trend<br />

## price_plots.py
Itt vizualizálom az ár adatokat és a létrehozott trendet.<br />

Fontosabb kimenete:<br />
Bitcoin_ar.png - A letöltés során a Bitcoin ára<br />
ar_elso_4000.png - A letöltött első 4000 áradat.<br />
trend_elso_4000.png - Az első 4000 adathoz tartozó trend<br />
## Download.py

A script ami letölti & feldolgozza, majd csv formátumba elmenti az orderbook-okat és btc price adatokat
