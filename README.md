# Sistem preporučivanja filmova korišćenjem podataka sa Letterboxd naloga

**Autor:** Pavle Sekulić

---

## 1. Ideja

[cite_start]Sistemi za preporučivanje filmova danas predstavljaju ključni deo digitalnih platformi za zabavu, jer korisnicima omogućavaju da lako pronađu sadržaj koji odgovara njihovom ukusu[cite: 4]. [cite_start]Ipak, većina ovih sistema zahteva velike količine direktnih podataka o ponašanju korisnika (pregledi, ocene, klikovi), što otežava personalizaciju bez prethodnog intenzivnog korišćenja sistema[cite: 5].

[cite_start]Iz tog razloga, u okviru ovog projekta razvijam sistem preporuka filmova koji koristi javno dostupne podatke sa Letterboxd naloga korisnika kao osnovu za kreiranje njihovog profila ukusa[cite: 6]. [cite_start]Letterboxd je filmska društvena mreža koja korisnicima omogućava da vode evidenciju o odgledanim filmovima, da ih ocenjuju, rangiraju i sve u svemu isprate pogledane filmove[cite: 7].

[cite_start]Cilj projekta je da na osnovu istorije gledanja i ocena sa korisnikovog Letterboxd naloga sistem generiše personalizovane preporuke koristeći tehnike mašinskog učenja i da te preporuke posle obradi i definiše da li su relevantne[cite: 8].

> [cite_start]**Inspiracija za projekat:** Ideju za predlog projekta sam dobio tako što sam uvideo mane u svojoj watchlisti za filmove (koja je samo jedan tekstualni fajl sa tristotinjak filmova u kom obeležavam jesam li pogledao film), tj. da ne mogu da znam o filmu ništa sem naslova bez da proverim na internetu[cite: 9, 10]. [cite_start]Sa ovim nedostacima, ja nikad nisam siguran koji film da pogledam, jer ne znam da li će mi se svideti[cite: 11].

---

## 2. Referentni radovi

[cite_start]Tokom istraživanja naišao sam na pregršt radova na ovu ili slične teme, ali sam se ograničio na desetak koje ću koristiti[cite: 13]:

* [cite_start]**Prvi rad [2]** predstavlja pregled metoda faktorizacije matrica za preporučivačke sisteme i uvodi osnove modela zasnovanih na latentnim faktorima, kao što su SVD i njihove varijante optimizovane stohastičkim gradijentnim spustom ili ALS metodom[cite: 14]. [cite_start]Takođe pominje mogućnost menjanja ulaza kroz vreme, kako preferencije evoluiraju[cite: 15].
* [cite_start]**Drugi rad [3]** objašnjava moguće probleme pri radu sa sistemima za preporučivanje (pominje *cold start* problem kao bitan) i bolje predstavlja metode rada poput kolaborativnog filtriranja, hibridnog filtriranja, i sl[cite: 16]. [cite_start]Takođe, pominje različite načine izračunavanja sličnosti dva entiteta (filma, korisnika) koji verovatno mogu promeniti uspeh modela[cite: 17]. [cite_start]Čini se veoma korisno jer pokriva i moguće probleme i prilike za unapređenje koncepta[cite: 18].
* [cite_start]**Ostali radovi** služe samo kao teorijska potpora ili interesantna mogućnost unapređenja[cite: 19].

---

## 3. Dataset

[cite_start]Dataset koji ću koristiti i koji se uglavnom koristi za slične modele jeste javno dostupni **MovieLens 32M** jer je noviji (podaci su od 1995. do 2024. godine) i ima dovoljno podataka[cite: 22]. Dataset sadrži:

* [cite_start]**32 miliona** ocena [cite: 23]
* [cite_start]**2 miliona** tagova [cite: 23]
* [cite_start]**88.000** filmova [cite: 23]
* [cite_start]**200.000** korisnika na sajtu IMDb [cite: 23]

[cite_start]Kako bih dobio informacije o korisnicima Letterboxda (korisniku samog sistema), iskoristio bih mogućnost preuzimanja podataka sa samog sajta u vidu `.csv` tabela sa pogledanim i ocenjenim filmovima[cite: 25]. [cite_start]Imajući u vidu problem *cold-starta*, morao bih da koristim korisničke naloge sa 20 ili više pogledanih, tj. ocenjenih filmova ili filmova stavljenih na watchlist[cite: 26, 27].

---

## 4. Metoda

[cite_start]Za treniranje modela bih koristio kombinaciju **Hibridnog filtriranja [5]** (što je samo po sebi kombinacija kolaborativnog filtriranja i filtriranja na bazi sadržaja) i **genetskog algoritma** koji bi određivao značaj pojedinih rezultata [rezultat CF-a, rezultat klasifikatora, i sl.](cite: 30).

* [cite_start]**Kolaborativno filtriranje (CF) [7, 8]:** Predstavlja način predviđanja rezultata na osnovu mišljenja drugih ljudi[cite: 32].
* [cite_start]**Klasifikator:** Uradio bih pomoću SVM algoritma i K-means grupisanja[cite: 33].
* [cite_start]**Filtriranje na osnovu sadržaja (content-based) [9]:** Metoda koja korisnicima predlaže predmete slične onima s kakvim su ranije interagovali[cite: 33]. [cite_start]Ono je važno zbog problema *cold-starta*[cite: 34]. [cite_start]Kao meru sličnosti koristio bih kosinusnu sličnost[cite: 35].
* [cite_start]**Genetski algoritam:** Koristio bi se kako bi maksimizovao preciznost i normalizovanu kumulativnu dobit[cite: 36].

[cite_start]Nakon treniranja klasifikatora, projektovao bih korisnički profil kroz transkripciju svih korisnikovih filmova sa karakteristikama izvučenim iz baze podataka [srednja vrednost tih vektora zajedno sa ocenama se čini kao moguće rešenje](cite: 37). [cite_start]U slučaju *cold-start* problema, razmislio bih o korišćenju podataka drugih korisnika koje korisnik prati na Letterboxdu i nekoj *neighbour-based* metodi[cite: 38]. [cite_start]Sve ove podatke mogu da dobijem pomoću `letterboxdpy` Python biblioteke[cite: 39].

[cite_start]Na kraju bih uradio predviđanje ocena za svaki film u bazi, uklonio filmove koje je korisnik već pogledao i na izlaz poslao najbolje ocenjenih **10-30 filmova**[cite: 40].

---

## 5. Aparatura

[cite_start]Nisam siguran da će na mom laptop računaru moći efikasno da se izvršava treniranje modela, zbog nedostatka grafičke kartice[cite: 42]. [cite_start]Ako bih se povezao sa mojim kućnim računarom ili namestio virtualnu mašinu, mislim da bi to bilo dovoljno[cite: 43].

---

## 6. Metrike

[cite_start]Koristiću standardne top-K preporučivačke metrike које су индустријски стандард[cite: 46]:

* [cite_start]**Precision@K** - Procenat filmova u K predloženih koji su zapravo relevantni [cite: 47]
* [cite_start]**Recall@K** - Procenat relevantnih filmova koji su predloženi [cite: 48]
* [cite_start]**Hit Rate** - Postojanje bar jednog relevantnog filma među K predloženih [cite: 49]
* [cite_start]**Normalizovana kumulativna dobit (NDCG)** - Vrednuje preporuke tako što daje veću težinu filmovima koji se nalaze visoko u listi [cite: 50]
* [cite_start]**Prosečna poziciona preciznost** - Prosečna preciznost preko celog skupa korisnika [cite: 51]

---

## 7. Formalni matematički model

Neka je dato konačno skup korisnika:
[cite_start]$$U=\{u_{1},u_{2},...,u_{|U|}\}$$ [cite: 54]

i skup filmova:
[cite_start]$$I=\{i_{1},i_{2},...,i_{|I|}\}$$ [cite: 56]

Ocene korisnika nad filmovima predstavljene su delimično popunjenom matricom:
[cite_start]$$R\in\mathbb{R}^{|U|\times|I|}$$ [cite: 58]

[cite_start]gde element $T_{ui}$ označava ocenu koju je korisnik $u$ dodelio filmu $i$, dok je $r_{ui}$ nedefinisano ukoliko film nije ocenjen[cite: 59, 61].

### 7.1 Kolaborativno filtriranje

[cite_start]U okviru kolaborativnog filtriranja primenjuje se model faktorizacije matrice ocena, pri čemu se matrica $R$ aproksimira proizvodom dve matrice manjeg ranga[cite: 62]:
[cite_start]$$R\approx \hat{R}=PQ^{T}$$ [cite: 64]

gde je:
[cite_start]$$P\in\mathbb{R}^{|U|\times k}$$ [cite: 65]
[cite_start]$$Q\in\mathbb{R}^{|I|\times k}$$ [cite: 66]

[cite_start]a $k\ll \min(|U|,|I|)$ predstavlja veličinu latentnog prostora u kojem radimo[cite: 67]. Predviđena ocena korisnika $u$ za film $i$ data je izrazom:
[cite_start]$$\hat{r}_{ui}=p_{u}^{T}q_{i}$$ [cite: 69]

[cite_start]gde su $p_{u}$ i $q_{i}$ latentni vektori korisnika i filma, respektivno[cite: 70]. Model se trenira minimizacijom regularizovane kvadratne greške:
[cite_start]$$\min_{P,Q}\sum_{(u,i)\in\Omega}(r_{ui}-p_{u}^{T}q_{i})^{2}+\lambda(\sum_{u}||p_{u}||^{2}+\sum_{i}||q_{i}||^{2})$$ [cite: 73]

[cite_start]gde $\Omega$ predstavlja skup poznatih ocena, a $\lambda$ je kazneni parametar za regulaciju[cite: 74].

### 7.2 Filtriranje zasnovano na sadržaju (Content Based)

Svaki film $i\in I$ opisan je vektorom karakteristika:
[cite_start]$$x_{i}\in\mathbb{R}^{d}$$ [cite: 77]

[cite_start]koji obuhvata informacije o žanrovima, tagovima i ostalim metapodacima[cite: 78]. Korisnički profil $c_{u}$ konstruiše se kao težinska srednja vrednost karakteristika filmova koje je korisnik ocenio:
[cite_start]$$c_{u}=\frac{1}{\sum_{i\in I_{u}}r_{ui}}\sum_{i\in I_{u}}r_{ui}\cdot x_{i}$$ [cite: 80]

[cite_start]gde $I_{u}$ predstavlja skup filmova koje je korisnik $u$ ocenio[cite: 81]. Sličnost između korisnika $u$ i filma $i$ računamo korišćenjem kosinusne sličnosti:
[cite_start]$$s_{ui}^{CB}=\cos(c_{u},x_{i})=\frac{c_{u}\cdot x_{i}}{||c_{u}||||x_{i}||}$$ [cite: 84]

### 7.3 Hibridni model

[cite_start]Hibridni sistem kombinuje rezultate kolaborativnog i sadržajnog filtriranja linearnom kombinacijom[cite: 85]:
[cite_start]$$s_{ui}^{H}=\alpha\cdot\hat{r}_{ui}+(1-\alpha)\cdot s_{ui}^{CB}$$ [cite: 86]

[cite_start]gde je $\alpha\in[0,1]$ težinski koeficijent koji određuje relativni doprinos pojedinih komponenti[cite: 87].

### 7.4 Optimizacija pomoću genetskog algoritma

[cite_start]Parametar $\alpha$, kao i eventualni dodatni parametri modela, optimizuju se pomoću genetskog algoritma[cite: 89]. [cite_start]Svaka jedinka u populaciji predstavlja vektor parametara[cite: 90]:
[cite_start]$$\theta=(\alpha,\lambda,k,...)$$ [cite: 91]

[cite_start]Fitness funkciju definišemo kao linearnu kombinaciju evaluacionih metrika[cite: 92]:
[cite_start]$$F(\theta) = \beta_1 \cdot \text{NDCG}@K + \beta_2 \cdot \text{Precision}@K$$ [cite: 93]

[cite_start]gde su $\beta_{1}$ i $\beta_{2}$ težinski koeficijenti[cite: 94]. [cite_start]Cilj genetskog algoritma je nalaženje parametara[cite: 95]:
[cite_start]$$\theta^{*}=\arg\max_{\theta}F(\theta)$$ [cite: 96]

tj. parametra gde je vrednost funkcije najveća[cite: 97].

### 7.5 Generisanje preporuka

Za svakog korisnika $u$ formira se rang-lista filmova[cite: 100]:
$$L_{u}=\{i\in I\backslash I_{u}\}$$ [cite: 101]

sortirana opadajuće po vrednosti $s_{ui}^{H}$[cite: 102]. Konačna preporuka sastoji se od prvih K filmova sa liste $L_{u}$[cite: 102].

---

## 8. Potencijalna unapređenja

* [cite_start]**Duboke neuronske mreže:** Primena NCF modela [8] za nelinearno učenje latentnih reprezentacija[cite: 104].
* [cite_start]**Vremenska dimenzija:** Uvođenje sekvencijalnih modela kao što su RNN-ovi ili Transformeri [10] radi praćenja promenljivosti preferencija[cite: 105, 106].
* [cite_start]**NLP metode:** Integracija tekstualnih recenzija i opisa filmova za obogaćivanje reprezentacija semantičkim informacijama[cite: 107].
* [cite_start]**Napredna optimizacija:** Zamena genetskog algoritma drugim poluheurističkim metodama poput PSO algoritma [12] radi brže konvergencije[cite: 108].
* [cite_start]**Korisnički interfejs:** Razvoj interaktivnog interfejsa za prikupljanje povratnih informacija у реалном времену[cite: 109].

> [cite_start]Većina ovih ideja nije izvodljiva na kampu od dve nedelje, ali služe kao dobar podsetnik da je svaki projekat moguće unaprediti[cite: 110].

---

## 9. Reference / Literatura

* **[1]** Francesco Ricci, Lior Rokach i Bracha Shapira. „Introduction to recommender systems handbook". U: *Recommender Systems Handbook* (2011.), str. [cite_start]1-35[cite: 114].
* **[2]** Yehuda Koren, Robert Bell i Chris Volinsky. „Matrix factorization techniques for recommender systems". U: *Computer* 42.8 (2009.), str. [cite_start]30-37[cite: 115, 116].
* **[3]** Mahesh Goyani i Neha Chaurasiya. „A Review of Movie Recommendation System: Limitations, Survey and Challenges". U: *ELCVIA Electronic Letters on Computer Vision and Image Analysis* 19.3 (2020.), str. [cite_start]18-33[cite: 117].
* **[4]** F. Maxwell Harper i Joseph A. Konstan. „The MovieLens Datasets: History and Context". U: *Proceedings of the 24th International Conference on Intelligent User Interfaces.* 2015., str. [cite_start]19-28[cite: 119].
* **[5]** Robin Burke. „Hybrid recommender systems: Survey and experiments". U: *User Modeling and User-Adapted Interaction* 12.4 (2002.), str. [cite_start]331-370[cite: 129].
* **[6]** S. Agrawal i P. Jain. „An improved approach for movie recommendation system". U: *2017 International Conference on I-SMAC (IoT in Social, Mobile, Analytics and Cloud) (I-SMAC).* IEEE. 2017., str. [cite_start]336-342[cite: 130, 131].
* **[7]** J. Ben Schafer i dr. „Collaborative filtering recommender systems". U: *The Adaptive Web* (2007.), str. [cite_start]291-324[cite: 132].
* **[8]** Xiangnan He i dr. „Neural Collaborative Filtering". U: *Proceedings of the 26th International Conference on World Wide Web.* International World Wide Web Conferences Steering Committee. 2017., str. [cite_start]173-182[cite: 133].
* **[9]** Robin van Meteren i Martin van Someren. „Using Content-Based Filtering for Recommendation". U: *Proceedings of the Machine Learning in the New Information Age: MLnet/ECML 2000 Workshop* 30 (2000.), str. [cite_start]47-56[cite: 134].
* **[10]** Wang-Cheng Kang i Julian McAuley. „Self-Attentive Sequential Recommendation". U: *Proceedings of the IEEE International Conference on Data Mining (ICDM)* (2018.), str. [cite_start]197-206[cite: 135, 136].
* **[11]** Jacob Devlin i dr. „BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". U: *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics* (2019.), str. [cite_start]4171-4186[cite: 137, 138].
* **[12]** James Kennedy i Russell Eberhart. „Particle Swarm Optimization". U: *Proceedings of the IEEE International Conference on Neural Networks* (1995.), str. [cite_start]1942-1948[cite: 139].
