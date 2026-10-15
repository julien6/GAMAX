# Optimiseur conservateur GAMAX

Le compilateur dispose d'une couche distincte d'optimisation de son IR typée, avant génération JAX. Elle est indépendante des noms et chemins des modèles. L'API accepte `compile_model(..., optimization="none" | "safe")` et la CLI `--optimization none|safe`. Le mode par défaut reste `none`, pour conserver le code de référence. Chaque manifeste contient les passes appliquées, leur nombre de sites et leurs conditions de validité. Une compilation réussie ne certifie pas la fidélité d'une transformation.

La première passe évite l'évaluation des requêtes spatiales `population overlapping each` dans les entrées absentes ou masquées d'un `accumulate`. Elle utilise un branchement JAX compilé, conserve l'ordre des entrées présentes, les doublons, les capacités, les erreurs sur les branches actives et les tirages aléatoires conditionnels. Elle ne s'applique pas aux collections retournées par des actions/getters. Elle ne réordonne pas les agents, ne change pas le générateur aléatoire et n'assouplit aucune règle flottante. Sur les six sources de développement, elle transforme SIR ; les cinq autres modules restent identiques.

L'audit [des six modules](../results/optimizer-six-code-audit-2026-10-13/summary.json) confirme que le mode `none` émet les mêmes bytes que les modules précédemment acceptés contre GAMA. Les [tests de la passe](../tests/test_optimizer.py) les tests CLI et les primitives natives de sources complètes passent (17 tests distincts). La validation native du code transformé compare tous les états déclarés et toutes les identités depuis l'initialisation jusqu'au cycle 100 aux traces GAMA archivées. Le runner vérifie aussi, lorsqu'il mesure les coûts, le résultat final du kernel compilé sur tout l'horizon. Il publie les temps de compilation/chauffe et trois répétitions synchronisées du calcul après chauffe ; ces mesures de kernel excluent observations, rendu et sauvegardes et ne remplacent pas une campagne de coûts complets en processus neufs.

```bash
.venv/bin/gamax compile chemin/modele.gaml -o generated/optimized \
  --experiment Simulation --capacities '{"Host":1024}' --optimization safe
.venv/bin/python scripts/validate_optimizer.py \
  -o results/optimizer-new-check --cases sir_spatial --seeds 2
# Contrôle cycle par cycle sur dix seeds, sans microbenchmark :
.venv/bin/python scripts/validate_optimizer.py \
  -o results/optimizer-ten-seeds --cases sir_spatial \
  --seeds 1 2 3 4 5 6 7 8 9 10 --fidelity-only
```

La campagne de référence `official-six-performance-2026-10-13-v2` a été arrêtée avant modification du compilateur ; son statut conserve cette raison et les mesures déjà produites restent archivées. Toute nouvelle campagne doit figer la nouvelle empreinte, identifier le mode d'optimisation et renouveler les certificats correspondants. Il n'y a pas encore de choix automatique du programme le plus rapide, de réactivation de la fusion flottante, de spécialisation aux paramètres d'expérience ni d'optimisation générale de tous les ABM. Ces extensions devront chacune fournir une condition de sûreté, des contrôles différentiels et des mesures.

## Premier microbenchmark terminé

Le [contrôle SIR seed 2](../results/optimizer-sir-native-check-2026-10-13/summary.json) passe pour les modes `none` et `safe` : 101 états et identités comparés au moteur natif archivé, puis état final du kernel compilé sur 100 cycles comparé à nouveau. Les trois durées de calcul après chauffe sont :

| Mode | Répétitions, secondes | Médiane |
|---|---|---:|
| `none` | 98,72 ; 50,39 ; 59,55 | 59,55 s |
| `safe` | 48,07 ; 43,77 ; 47,81 | 47,81 s |

Le rapport des médianes vaut 1,245, soit environ 19,7 % de réduction du calcul JAX dans cet essai. Ce résultat est indicatif : un seed, trois répétitions dans un processus, mesures successives sur une machine partagée et variabilité élevée de la référence. Il ne certifie ni un gain robuste, ni une accélération par rapport à GAMA, ni un gain conjoint CPU/mémoire. Le [contrôle différentiel des dix seeds](../results/optimizer-sir-ten-seeds-2026-10-13/summary.json) complète la validation fonctionnelle sans microbenchmark ; consulter son champ `complete` avant de conclure.

## Validation fonctionnelle terminée

Le contrôle SIR optimisé est **terminé et passe sur seeds 1–10, 100 cycles**, soit 101 états par seed et **52 549 290 valeurs comparées** au moteur natif archivé. Les états discrets et les identités concordent exactement ; l'écart flottant maximal observé est **0**. Cette preuve concerne cette source complète, ses paramètres/capacités validés et cette transformation. Les cinq autres modules optimisés sont byte-identiques aux modules précédemment acceptés. Le [contrôle autonome](../results/optimizer-sir-standalone-2026-10-13/result.json) passe aussi sur seed 2/100 cycles dans le Python isolé sans GAMAX installé.

Les contrôles ne démontrent pas la sûreté de transformations futures, la compatibilité avec tout GAML, ni une supériorité de performance sur GAMA. Le mode optimisé reste explicite et les mesures complètes contre GAMA devront être relancées avec de nouveaux certificats et une nouvelle empreinte figée.

## Deuxième étape : réductions en flux et Mersenne Twister par blocs

Le mode `safe` courant ajoute deux transformations génériques. La première fusionne `accumulate` avec son consommateur lorsque l'analyse prouve des sommes indépendantes sur des compteurs locaux int32, sans écriture persistante, somme flottante ni dépendance au compteur dans les prédicats/incréments. Les incréments flottants et les lectures de getters sont exclus. Les sous-collections sont consommées dans l'ordre ; les requêtes gardent leur consommation aléatoire. Les doublons sont comptés et le débordement int32 conserve l'arithmétique modulaire. Pour SIR, cette passe supprime le tableau intermédiaire de 62 464 références et masques : elle traite des sous-collections de capacité 1 024. La borne du parcours respecte les trous du masque et élimine ses seules entrées finales absentes. Le [contrôle de cette étape](../results/optimizer-stream-sir-ten-seeds/summary.json) passe les dix seeds, mais la première paire de coût contre GAMA restait plus lente ; les [mesures](../results/optimizer-stream-versus-gama/summary.json) sont conservées.

La seconde transforme le renouvellement MT19937 en trois blocs de mots `[0,227)`, `[227,454)`, `[454,623)`, puis traite le mot 623. Dans chaque bloc, les références déplacées de 397 mots désignent un suffixe ancien ou un bloc déjà mis à jour. Le dernier mot utilise le mot 0 **déjà mis à jour**, comme la récurrence séquentielle. Les XOR, décalages et masques restent uint32 ; seed, tempering, index, ordre et nombre de tirages, ainsi que conversion float64, restent identiques. Il s'agit d'une réorganisation respectant les dépendances, pas d'un remplacement par le RNG JAX. Cette passe s'applique aux six modèles utilisant Mersenne, sans branche liée à leur nom.

Les [tests bit à bit](../tests/test_optimized_mersenne.py) comparent 45 états (seeds natives 1–10, motifs adversariaux et états aléatoires) sur douze renouvellements, puis 1 300 tirages franchissant plusieurs frontières de 624 mots. Les 22 tests ciblés passent (optimiseur, Mersenne, CLI et primitives natives). Les campagnes fonctionnelles du profil courant sont [SIR](../results/optimizer-block-sir-ten-seeds/summary.json) et [les cinq autres modèles](../results/optimizer-block-other-five/summary.json) ; vérifier leur champ `complete`. Les preuves et timings précédents du mode `safe` concernent leurs modules archivés, et non automatiquement ce nouveau profil.

Le runner de coûts accepte maintenant `--optimization safe`, `--cases` et `--fidelity-certificate`. Une transformation doit fournir le SHA du module, l'empreinte courante du compilateur, une preuve complète sur dix seeds/101 états et le hash de cette preuve. Le mode de référence doit toujours réémettre les bytes acceptés. La chauffe d'un kernel JAX fonctionnel conserve son état initial : cet état est réutilisé après chauffe, sans une seconde initialisation identique. Ce changement du protocole d'exécution est déclaré séparément des passes du compilateur ; les anciens timings complets ne doivent pas être assimilés aux nouveaux.

## Profil expérimental `fusion` et premiers gains sur GAMA

L'API/CLI accepte désormais `none`, `safe` et `fusion`. Ce dernier conserve les barrières arithmétiques et CPU fast-math désactivé, bloque `algsimp`, mais permet la fusion HLO. **Cette transformation doit être certifiée par comparaison native** : elle est rejetée sur Boids, et acceptée pour les cinq autres sources testées sur dix seeds/100 cycles. Elle ne devient pas le défaut. Le [rapport de coûts corrigé](optimizer_performance.md) montre des gains conjoints sur trafic et fourmis et une régression proies-prédateurs, avec rendu commun, sources intactes et sorties/pacing natifs désactivés explicitement.

Le cache RNG calcule les uniformes IEEE-754 exactement par paires à partir des mots MT ; il avance l'index seulement lorsqu'un tirage est consommé. L'avance groupée ne calcule pas les valeurs inutilisées et applique le nombre exact de renouvellements MT. Le helper général traite aussi les indices impairs. Les tests supplémentaires vérifient séquences, états, index et valeur suivante après des sauts de 0 à 1 300 uniformes et aux positions 0/1/622/623/624. L'état RNG privé comporte un cache float64 supplémentaire ; les interfaces publiques `reset`/`step` restent compatibles.


## Passes génériques supplémentaires (itération du 14 octobre)

Le profil opt-in possède désormais quatre autres transformations : maintien linéaire de l'ordre du quadtree lorsqu'une seule identité bouge (tri stable complet si les autres feuilles changent), sommes entières masquées parallèles avec débordement de même largeur, élimination des permutations spatiales observées uniquement par `length`/un comptage à prédicat pur de champs stockés, et court-circuit des réflexes de populations à garde fausse. Les permutations éliminées conservent les deux nombres de tirages des shuffles natifs via l'avance exacte du Mersenne. Les collections utilisées autrement gardent leur ordre original ; les réductions flottantes restent ordonnées.

Le court-circuit est testé contre GAMA natif sur scheduling, état global, valeurs par agent, variables locales et tirages conditionnels. Les nouvelles passes sont en validation sur les six sources officielles. Elles ne deviennent pas certifiées par leur seule compilation. Les coûts, preuves et rejets sont décrits dans [optimizer_performance.md](optimizer_performance.md).


## Profils et caches en cours de validation

`fusion-avx` est un profil supplémentaire de CPU : fusion HLO autorisée, fast-math et algsimp désactivés, ISA plafonnée à AVX. Il a passé les six sources/seeds 1–10/100 cycles dans la copie figée correspondante ; son pilote Boids accélère le calcul après chauffe sans accélérer le coût total sur 100 cycles. Ce résultat reste limité au matériel et au périmètre mesurés.

Les réductions flottantes peuvent compacter les indices sélectionnés tout en conservant exactement l'ordre des additions et les barrières. Le quadtree évite de recompter les feuilles ou de redistribuer une population sur des nœuds déjà subdivisés. Les resets déjà compilés peuvent recevoir l'état Mersenne en données au lieu de spécialiser la seed. Ces transformations ne rendent pas les initializers eager compilés par défaut.

Le cache de contacts actuel mémorise une seule ligne de voisinage, de taille linéaire, pour les comptages/longueurs purs dans une population avec un rayon partagé. Une réutilisation exige la même identité, le même rayon et le même epoch ; mouvements, créations et morts changent l'epoch. Les prédicats relisent les attributs et les shuffles consomment toujours leurs tirages natifs. L'expérience dense quadratique reste archivée, mais n'est plus émise en raison de sa régression de calcul. L'ordre du quadtree est réparé à chaque mort avant les insertions suivantes.

Une nouvelle simplification générique évite le parcours d'insertion du quadtree lorsque la capacité est au plus 100. Retirer l'identité avant insertion laisse alors moins de 100 occupants : le seuil natif de subdivision ne peut être atteint. La racine reste donc non subdivisée ; les feuilles et timestamps d'insertion restent identiques. Cette passe nécessite ses tests et une nouvelle validation des modules concernés.

Les profils `safe` et `fusion-avx` échouent sur Boids seeds 1 et 7 à 1 000 cycles. Aucun certificat ni résultat favorable de performance n'est attribué à cet horizon ; les tolérances ne sont pas relâchées.

Les compteurs de feuille du quadtree remplacent le recomptage de toute la population lors d'une insertion. Ils sont décrémentés avant une réinsertion ou une mort, redistribués avec les identités à une subdivision et incrémentés après insertion. Les tests comparent ces compteurs aux histogrammes exacts des feuilles sur naissances, mouvements et morts, y compris le seuil de capacité 100/101.

Une investigation de Boids à 1 000 cycles a isolé une différence d'un bit entre le sinus JAX et le sinus Java/GAMA d'un déplacement (cycle 835 du seed 1). Une spécialisation fdlibm pour les angles de déplacement normalisés est en validation. Le test couvre 4 036 angles et sinus/cosinus, bit à bit contre `StrictMath`, avec deux configurations XLA. Ce test ne prouve pas l'identité avec tous les intrinsics `Math` d'une JVM ; les trajectoires natives restent le critère d'acceptation. Les angles hors d'un tour conservent le chemin JAX et nécessitent leur propre validation.

## Initialisation et profils expérimentaux récents

`fusion-avx-host` ajoute une initialisation fonctionnelle sur NumPy, puis conserve les états et transitions JAX. Elle réutilise le code Python émis dans un namespace privé ; elle n'interprète pas la source GAML et ne nécessite aucun paquet GAMAX à l'exécution. Le reset public reste une opération sur l'hôte, comme la conversion native du seed déjà présente. La géométrie ordonnée des voisinages de grille est précalculée sous un budget fixe, tandis que les permutations et leurs tirages restent dynamiques. Les shuffles implicites jetés consomment exactement les mêmes mots MT. Les expressions d'initialisation de grille invariantes et sans effets peuvent être évaluées une fois puis diffusées. La [version figée](../results/optimizer-host-frozen-workspace/freeze.json) passe les six sources/dix seeds/100 cycles et Boids/dix seeds/1 000 cycles ; ses coûts restent limités aux cellules réellement mesurées.

`fusion-avx-batch` est un prototype supplémentaire pour les getters globaux entiers/booléens en lecture seule. La classification exclut actions, aléa utilisé, réductions flottantes et expressions non couvertes. Seuls les tirages de permutations non observées sont budgétés puis avancés en groupe ; les erreurs restent signalées. Ce profil nécessite une nouvelle preuve native et ne remplace pas les profils certifiés.

Le [rapport courant](optimizer_performance.md) distingue les copies figées, contrôles, régressions et pilotes. La compatibilité fonctionnelle et les gains restent limités à leurs paramètres, capacités, seeds, horizons et configurations JVM/JAX.


## Initialisations pures de grandes grilles et protocole amorti

Les profils batch expérimentaux émettent des fonctions JAX de cellule réutilisables pour les initialisations pures des grilles d'au moins 1 024 cellules. La lecture de champs stockés de cette cellule et de globals est autorisée ; getters, actions, RNG et autres-cellules sont exclus. La géométrie de grille et les branches conditionnelles couvertes restent contrôlées par l'IR. Le contexte NumPy privé appelle les fonctions compilées puis récupère des tableaux fonctionnels ; il ne change aucun binding du module. Les initialisations constantes à bornes invariantes restent diffusées, indépendamment des expressions de mise à jour qui s'exécuteront lors des cycles.

La version [figée](../results/optimizer-pure-grid-frozen-workspace/freeze.json) passe les [six sources sur dix seeds](../results/optimizer-pure-grid-six-ten-seeds/summary.json) et porte un [certificat lié aux émissions](../results/optimizer-pure-grid-six-certificate/certificate.json). Cela ne prouve ni un avantage à froid universel ni la compatibilité de tout GAML. Les codes des profils rapides/algébriques/à appels partagés restent expérimentaux et leurs mesures défavorables sont conservées.

Le runner [benchmark_amortized.py](../scripts/benchmark_amortized.py) complète les processus neufs par des batches de simulations. La JVM et l'expérience native compilée, ou l'exécutable JAX, sont réutilisés. Les coûts initiaux ne disparaissent pas : compilation/chauffe et total du batch sont rapportés. La création native répétée est d'abord comparée à chaque cycle aux oracles, avec seeds différentes puis répétées. Les garde-fous vérifient le code, les capacités, l'expérience, l'horizon, les dix seeds certifiées et la version de l'adaptateur natif. Les comparaisons de sorties conditionnent tous les ratios favorables.


## Garde de références de grille et scalaires NumPy

Le profil batch évite les vues de sous-classes pour les scalaires et tableaux NumPy de dimension zéro pendant l'initialisation. Les tableaux indexés restent fonctionnels : leurs mises à jour copient le buffer. Le [contrôle des 60 états privés](../results/optimizer-host-scalar-initialization/summary.json) compare types, valeurs et RNG bit à bit.

Une boucle de sommes int32 indépendantes sur `population overlapping cell` peut utiliser des histogrammes de références stockées vers cette grille. Le garde vérifie la géométrie rectiligne finie et séparée, la validité des références/cellules et l'alignement des agents vivants. Les multiplicités des cellules répétées, le débordement modulaire et chaque budget de tirages sont conservés. Les prédicats lisent seulement des valeurs stockées ; getters, RNG utilisé et dépendances entre accumulateurs sont exclus. Une condition non vérifiée exécute la boucle géométrique originale. Les fonctions restent déterminées par l'IR et les données, sans branche par nom ou chemin d'ABM.

Le [certificat courant des six émissions](../results/optimizer-grid-counts-final-six-certificate/certificate.json) est lié aux parents de preuve et aux hashes exacts du code réémis. Les résultats [SIR finaux](../results/optimizer-grid-counts-finite-sir-amortized-ten-seeds/analysis.json) couvrent dix seeds et trois répétitions sur 100 cycles, avec et sans rendu commun, en réutilisant les deux moteurs. Ils ne remplacent pas les coûts à froid ni les autres tailles/horizons.
