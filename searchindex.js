Search.setIndex({docnames:["accessing-data","api","api-define","api-errors","api-testing","api-utils","changelog","combine-tables","contributing","create-database","data-conventions","data-example","data-format","data-header","data-introduction","data-tables","emodb-example","genindex","index","install","map-scheme","update-database"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["accessing-data.rst","api.rst","api-define.rst","api-errors.rst","api-testing.rst","api-utils.rst","changelog.rst","combine-tables.rst","contributing.rst","create-database.rst","data-conventions.rst","data-example.rst","data-format.rst","data-header.rst","data-introduction.rst","data-tables.rst","emodb-example.rst","genindex.rst","index.rst","install.rst","map-scheme.rst","update-database.rst"],objects:{"":{audformat:[1,0,0,"-"]},"audformat.Column":{get:[1,2,1,""],rater:[1,3,1,""],rater_id:[1,4,1,""],scheme:[1,3,1,""],scheme_id:[1,4,1,""],set:[1,2,1,""],table:[1,3,1,""]},"audformat.Database":{__contains__:[1,2,1,""],__getitem__:[1,2,1,""],__setitem__:[1,2,1,""],author:[1,4,1,""],drop_files:[1,2,1,""],drop_tables:[1,2,1,""],expires:[1,4,1,""],files:[1,3,1,""],is_portable:[1,3,1,""],languages:[1,4,1,""],license:[1,4,1,""],license_url:[1,4,1,""],load:[1,2,1,""],load_header_from_yaml:[1,2,1,""],map_files:[1,2,1,""],media:[1,4,1,""],name:[1,4,1,""],organization:[1,4,1,""],pick_files:[1,2,1,""],pick_tables:[1,2,1,""],raters:[1,4,1,""],root:[1,3,1,""],save:[1,2,1,""],schemes:[1,4,1,""],segments:[1,3,1,""],source:[1,4,1,""],splits:[1,4,1,""],tables:[1,4,1,""],update:[1,2,1,""],usage:[1,4,1,""]},"audformat.Media":{bit_depth:[1,4,1,""],channels:[1,4,1,""],format:[1,4,1,""],sampling_rate:[1,4,1,""],type:[1,4,1,""],video_channels:[1,4,1,""],video_depth:[1,4,1,""],video_fps:[1,4,1,""],video_resolution:[1,4,1,""]},"audformat.Rater":{type:[1,4,1,""]},"audformat.Scheme":{draw:[1,2,1,""],dtype:[1,4,1,""],is_numeric:[1,3,1,""],labels:[1,4,1,""],maximum:[1,4,1,""],minimum:[1,4,1,""],replace_labels:[1,2,1,""],to_pandas_dtype:[1,2,1,""]},"audformat.Split":{type:[1,4,1,""]},"audformat.Table":{__add__:[1,2,1,""],__getitem__:[1,2,1,""],__setitem__:[1,2,1,""],columns:[1,4,1,""],copy:[1,2,1,""],db:[1,3,1,""],df:[1,3,1,""],drop_columns:[1,2,1,""],drop_files:[1,2,1,""],drop_index:[1,2,1,""],ends:[1,3,1,""],extend_index:[1,2,1,""],files:[1,3,1,""],get:[1,2,1,""],index:[1,3,1,""],is_filewise:[1,3,1,""],is_segmented:[1,3,1,""],load:[1,2,1,""],media:[1,3,1,""],media_id:[1,4,1,""],pick_columns:[1,2,1,""],pick_files:[1,2,1,""],pick_index:[1,2,1,""],save:[1,2,1,""],set:[1,2,1,""],split:[1,3,1,""],split_id:[1,4,1,""],starts:[1,3,1,""],type:[1,4,1,""],update:[1,2,1,""]},"audformat.define":{DataType:[2,1,1,""],Gender:[2,1,1,""],IndexField:[2,1,1,""],IndexType:[2,1,1,""],License:[2,1,1,""],MediaType:[2,1,1,""],RaterType:[2,1,1,""],SplitType:[2,1,1,""],TableStorageFormat:[2,1,1,""],Usage:[2,1,1,""]},"audformat.define.DataType":{BOOL:[2,4,1,""],DATE:[2,4,1,""],FLOAT:[2,4,1,""],INTEGER:[2,4,1,""],STRING:[2,4,1,""],TIME:[2,4,1,""]},"audformat.define.Gender":{CHILD:[2,4,1,""],FEMALE:[2,4,1,""],MALE:[2,4,1,""],OTHER:[2,4,1,""]},"audformat.define.IndexField":{END:[2,4,1,""],FILE:[2,4,1,""],START:[2,4,1,""]},"audformat.define.IndexType":{FILEWISE:[2,4,1,""],SEGMENTED:[2,4,1,""]},"audformat.define.License":{CC0_1_0:[2,4,1,""],CC_BY_4_0:[2,4,1,""],CC_BY_NC_4_0:[2,4,1,""],CC_BY_NC_SA_4_0:[2,4,1,""],CC_BY_SA_4_0:[2,4,1,""]},"audformat.define.MediaType":{AUDIO:[2,4,1,""],OTHER:[2,4,1,""],VIDEO:[2,4,1,""]},"audformat.define.RaterType":{HUMAN:[2,4,1,""],MACHINE:[2,4,1,""],OTHER:[2,4,1,""],TRUTH:[2,4,1,""],VOTE:[2,4,1,""]},"audformat.define.SplitType":{DEVELOP:[2,4,1,""],OTHER:[2,4,1,""],TEST:[2,4,1,""],TRAIN:[2,4,1,""]},"audformat.define.TableStorageFormat":{CSV:[2,4,1,""],PICKLE:[2,4,1,""]},"audformat.define.Usage":{COMMERCIAL:[2,4,1,""],OTHER:[2,4,1,""],RESEARCH:[2,4,1,""],RESTRICTED:[2,4,1,""],UNRESTRICTED:[2,4,1,""]},"audformat.errors":{BadIdError:[3,1,1,""],BadTypeError:[3,1,1,""],BadValueError:[3,1,1,""]},"audformat.testing":{add_table:[4,5,1,""],create_audio_files:[4,5,1,""],create_db:[4,5,1,""]},"audformat.utils":{concat:[5,5,1,""],duration:[5,5,1,""],intersect:[5,5,1,""],join_labels:[5,5,1,""],join_schemes:[5,5,1,""],map_language:[5,5,1,""],read_csv:[5,5,1,""],to_filewise_index:[5,5,1,""],to_segmented_index:[5,5,1,""],union:[5,5,1,""]},audformat:{Column:[1,1,1,""],Database:[1,1,1,""],Media:[1,1,1,""],Rater:[1,1,1,""],Scheme:[1,1,1,""],Split:[1,1,1,""],Table:[1,1,1,""],assert_index:[1,5,1,""],define:[2,0,0,"-"],errors:[3,0,0,"-"],filewise_index:[1,5,1,""],index_type:[1,5,1,""],segmented_index:[1,5,1,""],testing:[4,0,0,"-"],utils:[5,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","property","Python property"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:property","4":"py:attribute","5":"py:function"},terms:{"0":[0,1,2,5,7,9,10,11,15,16,20],"00":[0,1,5,7,10,11,15,16],"000":11,"000975027":11,"001":[0,4,7,11,20,21],"001000":1,"002":[0,4,7,11,20,21],"0020313625115058187":11,"003":[7,11,20,21],"004":[7,11,20,21],"004318584":11,"005":[11,20,21],"006":11,"006250":16,"007":11,"008":11,"009":11,"01":[0,1,5,7,10,11,15,16],"010":11,"010000":11,"011":11,"011063614":11,"012":11,"013":11,"013456696":11,"014":11,"015":11,"016":11,"017":11,"018":11,"019":11,"01968926392165793":11,"02":[1,5,7,11,15,16],"020":11,"020000":11,"020242583":11,"021":11,"022":11,"022717177":11,"023":11,"024":11,"02427998985529256":11,"025":11,"026":11,"027":11,"028":11,"028004277":11,"028277270039894997":11,"029":11,"03":[5,7,11,15,16],"030":11,"030000":11,"030466":11,"030466224736303227":11,"031":11,"032":11,"033":11,"033049549":11,"033448517":11,"034":11,"03401135464430882":11,"035":11,"036":11,"037":11,"038":11,"039":11,"03a01fa":16,"03a01nc":16,"03a01wa":16,"03a02fc":16,"03a02nc":16,"03a02ta":16,"03a02wb":16,"03a02wc":16,"03a04ad":16,"03a04fd":16,"03b10ab":16,"04":[7,11],"040":11,"040000":11,"040902":7,"041":11,"041592":11,"042":11,"043":11,"044":11,"045":11,"046":11,"047":11,"048":11,"048458810107741446":11,"048849880":7,"049":11,"05":11,"050":11,"050000":11,"05095786265974889":11,"051":11,"052":11,"052400952":11,"0525333518931399":11,"053":11,"054":11,"055":11,"056":11,"057":11,"058":11,"059":11,"059235768211694184":11,"059602150":11,"06":11,"060":11,"060000":11,"061":11,"062":11,"063":11,"063500998":11,"064":11,"064441694":11,"065":11,"065820":7,"066":11,"067":11,"068":11,"069":11,"06946232564591481":11,"07":11,"070":11,"070000":11,"071":11,"072":11,"073":11,"074":11,"075":11,"076":11,"077":11,"07733221007650082":11,"078":11,"079":11,"07v5yb3q9j":11,"08":[11,16],"080":11,"080000":11,"081":11,"081249":7,"082":11,"08241753741729374":11,"083":11,"084":11,"085":11,"086":11,"086022099":11,"087":11,"088":11,"089":11,"09":[11,16],"090":11,"091":11,"09112550904868355":11,"092":11,"093":11,"093510049":11,"094":11,"095":11,"096":[11,21],"097":[11,21],"098":[11,21],"099":[11,21],"099573507":11,"09a07na":16,"0asvuuowh1":11,"0jrpas9pac":11,"0m":1,"0s":15,"0ugcr4noai":11,"1":[1,2,4,5,9,10,11,13,15,16,20,21],"10":[1,11,16,21],"100":[11,16,21],"1000":[1,10],"100000":[1,11],"10090482846496696":11,"101":21,"102":21,"102429355":11,"103":21,"104":21,"105":21,"107334641":11,"1089436983129699":11,"11":[11,16],"110000":11,"114136332":11,"11695052743529533":11,"117774786":11,"119850006":11,"12":[10,11,16],"120":11,"120258339":11,"123625":16,"12752359432153892":11,"13":[11,16],"130":[11,16],"130000":11,"13159338895686545":11,"131611362":11,"13278217551698868":11,"134518334":11,"135201345":11,"13646606585515197":11,"138509":7,"14":[11,16],"140":11,"14427148834001335":11,"146743":7,"148101":11,"14810132514388807":11,"14a04tc":16,"15":[11,16],"150":11,"150000":11,"150441957":7,"15758643225131563":11,"159971439":7,"16":[11,16],"160":11,"16000":[1,4,11,16],"160000":11,"164587665":11,"17":11,"171822333":11,"174303":11,"17430302842382983":11,"175666791":7,"178356128":11,"178403028":11,"17847537821565207":11,"178744347":11,"18":11,"180":11,"180000":11,"180138149":7,"180676100":11,"186232247":11,"187305972":11,"1893636378033654":11,"189364":11,"19":11,"190":11,"190000":11,"190869338":11,"1970":[0,11],"1997":16,"1999":16,"1b46pwiyph":11,"1h":15,"1k4l6y2s42":11,"1lo0kjvcit":11,"1m":[1,15],"1s":15,"1st":13,"1wdchodkkn":11,"2":[0,1,5,7,9,11,15,16,21],"20":[7,11,16,21],"200":11,"2000":1,"200000":11,"2024565297512817":11,"203660568":11,"20504518768766467":11,"205699481":11,"21":[11,16,21],"210":11,"210876524":11,"214461149":11,"215952632661492":11,"219703589":11,"22":11,"220":11,"220000":11,"220609591":11,"22561003140146108":11,"225904836":7,"2267036729815184":11,"227082418":11,"229869707":11,"23":[11,21],"230":11,"230000":11,"2311020181304737":11,"23419103478381764":11,"2354898692751568":11,"23601762336894572":11,"24":11,"240":11,"240000":11,"240031918":11,"24275985903937558":11,"243617881":11,"244581520":11,"24721402094386802":11,"24782070868503858":11,"24903422178704582":11,"24954068194953227":11,"25":[11,16],"250":11,"251662052":11,"251980459":11,"254659914":11,"2549584483826606":11,"257006300":11,"259241015":11,"26":[7,11,16],"260000":11,"26585281466357036":11,"27":11,"272014363":7,"274200285":11,"277770049":11,"277941425":11,"278344509":7,"279490977":11,"28":11,"280000":11,"28272177205673277":11,"2854979876896957":11,"285971203":11,"29":[11,21],"290":11,"290326094":11,"293984746":7,"29502171718502046":11,"297296119":11,"298167269":11,"29c9amiwk":11,"2fpkgbir1o":11,"2s":15,"2utmz1dl3v":11,"3":[1,5,7,9,11,13,15,16],"30":[11,16,20,21],"300000":11,"300442442":7,"301620042":11,"30659373786991706":11,"31":[10,11,16,21],"310":11,"311991473":11,"316xvzi01v":11,"32":[11,16,21],"320":11,"320000":11,"321547990":11,"324615398":11,"326121502":7,"326480":7,"327968621":11,"32d8cjd4pl":11,"33":[1,11,20],"330":11,"3303785022718966":11,"336902484":11,"34":[11,16],"340":11,"340000":11,"348862270":11,"35":[11,16],"350":11,"35092943386049114":11,"354422564":11,"36":[11,21],"360":11,"360000":11,"360004":7,"363928425":11,"365422982":11,"367361701":11,"367758":7,"37":[11,20,21],"370":11,"370000":11,"3705062036963762":11,"373384052":7,"3734006372783616":11,"373401":11,"375422060":11,"376569334820559":11,"376653":7,"376911426":11,"378495064":11,"378642596":11,"38":11,"380":11,"3819909942154569":11,"39":11,"390":11,"390000":11,"391645829":11,"39210425355097445":11,"396450028":11,"3k4bbs1som":11,"3n2jwn8y2z":11,"3s":15,"3vnwzwfww2":11,"4":[2,5,7,9,11,15,16],"40":[7,11],"4002406906544087":11,"408456186":11,"409496329":7,"41":[11,21],"410":11,"410035897":11,"41399087846643856":11,"41490077267604986":11,"4194926880104026":11,"42":[11,21],"421452378":11,"424504397":11,"427947269":11,"43":[7,11],"430000":11,"436196868":11,"436632930":11,"43952626489472224":11,"439812500":16,"44":[1,11,21],"440000":11,"441312":7,"4462094428404644":11,"44663718228191773":11,"449197669":11,"45":11,"450":11,"450000":11,"450071513":11,"45054701757392435":11,"4539202295078586":11,"45812623482790893":11,"459056943":7,"459561419":11,"459861623":11,"46":11,"460":11,"460000":11,"461309":7,"46159134270870505":11,"4622443560197509":11,"467224047":11,"467302208":11,"47":[11,21],"470":11,"470000":11,"470207452":11,"471623202":11,"476041998":7,"47646772984879737":11,"478328342":11,"4799218114702084":11,"48":11,"480000":11,"482416292":7,"485718":7,"49":11,"490":11,"490000":11,"491265315":11,"491370207":11,"491677107":11,"491775112":11,"4934035976031872":11,"494414126":11,"4949497524298141":11,"495870":7,"497456799":11,"498062500":16,"4984282548355199":11,"499389597":7,"4d685boala":11,"4dmkmw7oha":11,"4hdxwy3oz":11,"4lb6xv8l5m":11,"4uv9lnzcqh":11,"5":[1,4,5,9,15,16],"50":[7,11,21],"500":[10,16],"500000":[10,11],"501148602":11,"5022570967595972":11,"504874999":16,"509951222":7,"51":11,"510":11,"510000":11,"5109543252344082":11,"511095303":11,"511800940":11,"514730736":11,"515634666":11,"515819867":7,"517618531":11,"519964328":11,"52":11,"520":11,"520000":11,"5221741697235134":11,"523246917":11,"524218646":11,"5288923265727632":11,"5289915529572562":11,"53":11,"530":11,"530000":11,"5301155862301754":11,"532355960":11,"536095971":11,"538535889":11,"54":11,"541133953":11,"543207049":11,"545000677":7,"5462931264660738":11,"5474007976621189":11,"55":11,"550":11,"550000":11,"5506149039912587":11,"552376":11,"5523763316563404":11,"553346556":7,"553605060":11,"554626139":7,"554988998":7,"56":11,"560":11,"562008131":11,"56treu7w87":11,"57":11,"570":11,"571911674":7,"573543259":11,"575617942":7,"575871640553717":11,"5789649149004863":11,"58":11,"580000":11,"5876035841652747":11,"587855983":7,"589398144421158":11,"59":11,"590":11,"590000":11,"593138808":11,"5996684172624545":11,"5elyrktoei":11,"5fsb5tq8ot":11,"5g49aha7d8":11,"5qp2tbpkmk":11,"5s":4,"6":16,"60":[4,11],"600":[1,11],"600000":11,"600627257":11,"601261546379271":11,"604249571":11,"604651920":7,"609530916":11,"6099330449001787":11,"61":11,"610":11,"610000":11,"611250":16,"617022887":11,"6176488323630898":11,"620":11,"620000":11,"6205416680625613":11,"6210914649977601":11,"623224479":11,"623663425":7,"6247340378105861":11,"625348751":11,"63":11,"630000":11,"630986193721824":11,"633254073":11,"638488426":11,"6386158742428842":11,"638985149":11,"639":5,"64":11,"640":11,"640000":11,"648353013116614":11,"65":11,"653110800":11,"6560964560073761":11,"6580133330976448":11,"66":11,"660":11,"660000":11,"668584469":11,"668594726":11,"67":11,"670":11,"670000":11,"670368359":11,"6766948371446484":11,"676728699":11,"676980779":11,"6781166053610859":11,"6783482170579513":11,"679697093":11,"679772124":11,"68":11,"680":11,"680000":11,"681832055":11,"685342":7,"6884255554048742":11,"688821719307898":11,"689455541":11,"69":11,"690":11,"690000":11,"691ph3t0vi":11,"692816023":11,"693988031":11,"694339236":11,"695064703":11,"696812500":16,"6969638894607768":11,"696964":11,"698158479":7,"6iqofusfdd":11,"6ny7rfsdfr":11,"6snjogfqn":11,"6unhmot1fk":11,"6xol93hqrm":11,"7":11,"700":11,"700000":11,"701004840":11,"703396833":11,"703650473":11,"706575358":7,"71":11,"710":11,"710000":11,"713953285214048":11,"714979504":11,"7175118779953598":11,"72":11,"720":11,"720000":11,"7205038343954888":11,"720504":11,"724327779":11,"725803107722333":11,"726683040":11,"7267150092509207":11,"7272681226214479":11,"729668":7,"730000":11,"7309177564421498":11,"732208240":11,"733838":11,"7338383499963501":11,"735687500":16,"74":11,"740":11,"740000":11,"7424406488996806":11,"745059388":7,"750":11,"750000":11,"750645467":11,"751044392":11,"7512587077633304":11,"752329235468657":11,"754333960":11,"758047064":11,"758373848670072":11,"76":11,"760":11,"760000":11,"763839":11,"7638390379409621":11,"764527109":11,"765739262":7,"7675496183649375":11,"768368677747514":11,"7694478036370951":11,"769448":11,"769892904":7,"76uplhdlzi":11,"77":11,"770":11,"770000":11,"7792905337744622":11,"779488":11,"7794880478404012":11,"78":11,"780":11,"780000":11,"781882670":11,"7848837748661672":11,"787495":7,"788746100":11,"789677":7,"79":11,"790":11,"790000":11,"791084234":11,"792909":11,"7929090615030938":11,"793342788":11,"794257221134661":11,"794305737":11,"796382188":11,"7999731579348179":11,"7hg17kdv6k":11,"7nbrcpjwh4":11,"7q9rs5upz5":11,"7r8hgt9534":11,"8":[11,16],"80":11,"800":[1,11],"803276704":11,"803890216":11,"8049463908868857":11,"806804":11,"8068043268993625":11,"806893":7,"806916839":7,"8091968577132098":11,"8093955677841811":11,"809707312":11,"81":11,"810":[0,11],"810000":11,"813218061":11,"813870127":11,"816289559126766":11,"82":11,"823762338":11,"826118132":11,"826600580":11,"83":11,"830":11,"830000":11,"8334727859712343":11,"8362613143873328":11,"837421170":11,"84":11,"840":11,"840000":11,"841162206":11,"846771737":11,"85":[10,11,16],"850":11,"850000":11,"8506":16,"8582721693162672":11,"86":11,"860":11,"860000":11,"868914343":11,"8699218327373845":11,"87":11,"870":11,"870312786":11,"8704446608655305":11,"871882875":11,"874380743":11,"8769100225947914":11,"877812500":16,"8789867042050452":11,"88":11,"880":11,"880000":11,"881129064":11,"881889997":11,"882520398":11,"886180468":11,"8891174845855934":11,"889774877":11,"89":11,"890":11,"893798":7,"898250":16,"8987942213279687":11,"8kpvhmuhmt":11,"8l3sb7m6de":11,"9":[11,16],"90":[11,16],"900000":11,"9000824267670251":11,"903016":7,"903647855":7,"905470917":11,"906762474":11,"907276421":7,"91":11,"910":11,"910000":11,"915240919":11,"917030428":11,"9177168089378493":11,"918587212":11,"920":11,"920000":11,"922987517":11,"9258559809087041":11,"927657672":11,"9294314783194804":11,"930":11,"930000":11,"939107772":7,"94":11,"940":11,"940000":11,"942069941":7,"944640058":11,"9450904218667338":11,"9463181837112806":11,"9474713802180772":11,"9486288061890453":11,"949240":11,"9492402868531156":11,"95":[11,16],"950000":11,"951107314":7,"951820994":11,"9523098386464737":11,"96":11,"960000":11,"9666918445228314":11,"969898422":11,"97":11,"970":11,"973577776":11,"9773915965833093":11,"977885622":11,"978923720":11,"97myruoh5g":11,"98":11,"980":11,"985052421":11,"985221441":7,"985914170":11,"986067398":11,"9864215237495113":11,"986422":11,"987786464":11,"988870636":11,"989437636":11,"989929915":11,"99":11,"990":11,"990000":11,"993408502":11,"9943784456892997":11,"994815029":11,"997605326":11,"9993871018215263":11,"9eakjci47f":11,"9iezgztqdp":11,"9isu6xixwr":11,"9mmbk21qhr":11,"9uuwsdapca":11,"9vx8vawvsr":11,"boolean":6,"case":[1,10,14],"class":[1,2,3],"default":[1,4,5,13],"do":[0,1,5,6,13,20],"f\u00fcr":16,"final":21,"float":[1,2,4,5,6,7,9,10,11,13,16],"function":[1,4,16],"holzst\u00fcck":16,"import":[0,4,5,7,9,10,11,13,15,16,20,21],"int":[1,2,4,5,6,7,9,11,16,21],"k\u00f6nnte":16,"long":[1,10],"new":[1,5,10,21],"public":[2,6],"return":[0,1,4,5,6,20],"short":[0,13],"st\u00fcck":16,"static":1,"t\u00fcten":16,"true":[0,1,5,6,7,9,11,15,16,20,21],"try":1,A:[1,4,11,15,16,21],And:[7,20],As:16,BY:2,But:20,By:[1,4,5],For:[0,1,4,5,9,16,21],If:[1,4,5,7,8,9,10,20],In:[1,4,9,10,16,20],It:[7,8,9,10,15,16,20],Or:[0,7,9,10,21],That:7,The:[1,4,5,6,7,8,9,10,12,13,14,16,18,20,21],Their:16,Then:16,There:[1,15],These:13,To:[0,1,4,8,13,19,20,21],__add__:[1,6],__contains__:1,__eq__:6,__getitem__:1,__setitem__:1,a01:16,a02:16,a04:16,a05:16,a07:16,a7brmaymco:11,abend:16,abgeben:16,about:[7,12,13,16,20],absolut:[0,1,5],abspath:0,access:[10,12,13,14,15,16],accord:7,accur:10,acoust:16,across:[7,12,14],act:16,action:5,activ:[8,19],actor:16,actual:[12,20],ad:[1,5,6,10,21],add:[0,1,4,6,7,9,13,15,16,21],add_tabl:[7,20,21],addit:[1,9,10,13,16,20],adher:6,af:[10,16],after:[6,10],afterward:16,ag:[1,7,10,14,16,20,21],again:[6,21],agent:10,aggreg:[10,14],agn:16,agreement:16,all:[0,1,5,6,7,8,9,10,14,15,16],allow:[1,4,13,14],allow_nat:[5,6],also:[1,5,7,8,10,20,21],alwai:[1,8,10],am:16,amplitud:4,an:[1,3,5,8,10,12,13,14,15,16,18],anecho:16,anger:[9,16],ani:[1,3,20],annot:[0,1,8,9,12,14,18,21],anoth:[1,10],apcltxuxwa:11,appear:13,append:10,appli:[1,4,5,13,14],applic:14,ar:[1,4,5,6,7,8,10,12,13,15,16,20,21],arbitrari:[13,15],arg:5,argument:[5,6,10,16,20],around:9,arous:10,ask:[1,16],asr:10,assert:1,assert_index:6,assign:[1,6,13,15,16],associ:[1,10],assum:[9,10,21],astyp:16,attach:7,attribut:20,audeer:[1,8,9,10,11,13,14,16],audformat:[0,6,7,8,9,10,11,14,19,20,21],audio:[0,1,2,4,5,7,11,12,13,14,16,18,20,21],audiofil:[5,10,16],aueekdhjri:11,auf:16,author:[1,6,11],automat:[1,5,8,10],avail:[1,7,8,16],averag:[10,11],avi:11,avoid:6,ayppbtuzl4:11,azgctvnh3t:11,b01:16,b02:16,b03:16,b09:16,b10:16,b8pgkghpda:11,b:[1,5,8,10,11],badiderror:1,badvalueerror:1,bafuqqouw1:11,bar:[1,5],base:[1,6],basename_wo_ext:16,basic:16,bbqu4wdba8:11,becaus:1,been:[1,15],befindet:16,befor:16,begin:15,belong:[7,10,13,16],below:13,berlin:16,besucht:16,between:12,bilderbar:16,bin:[8,16,19],bit:[1,13],bit_depth:[1,11,13],bitmvcdpzr:11,blank:10,bool:[0,1,2,4,5,6,9,11],boredom:16,both:[1,5,7],bound:10,btwqrdket8:11,bygndb6ftf:11,c4f9h0jxq:11,c8hw3wbnth:11,c9aauy4jyt:11,c9llhs8qep:11,c:[1,5,11],calcul:[5,10],call:[1,4,5],callabl:[1,4],can:[0,1,4,5,7,8,9,10,12,13,14,15,16,18,20],cannot:[1,6],categor:14,categori:10,categoricaldtyp:1,caus:3,cc0:[2,11],cc0_1_0:2,cc:2,cc_by_4_0:2,cc_by_nc_4_0:2,cc_by_nc_sa_4_0:2,cc_by_sa_4_0:2,cd:8,cdurgebnb7:11,cgfpb76oki:11,challeng:14,chamber:16,chang:[1,6,8],changelog:8,channel:[1,4,11,13,16],charact:10,character:13,characterist:15,check:[1,6,8,16],child:2,citeseerx:16,classifi:[1,11],client:10,clone:8,code:[0,9,16],collect:[10,16,21],column:[0,2,4,5,6,7,9,10,11,12,15,16,20,21],column_id:1,columnid:13,com:[1,8,9,11,13],combin:[1,5,6,18],combined_t:7,come:15,command:9,commerci:[1,2,13],commit:8,common:[2,10],compare_2016:11,compress:6,concat:6,concaten:5,condit:1,confid:[1,14,16],conform:[1,5,6],connect:12,consid:[1,10,21],consist:[1,10,15],contain:[1,5,6,9,10,12,16,18,21],content:[2,10,12],continu:1,convers:5,convert:[0,1,5,6,12,16],copi:[1,7],copy_media:[1,6],copytre:16,core:[1,13],corpu:14,correct:[6,16],correspond:18,could:16,cover:9,cpdfk9xo8w:11,cpxz1i89zt:11,cqinnc9z5u:11,creat:[1,4,5,7,10,13,14,15,19],create_audio_fil:6,create_db:[0,7,9,11,15,20,21],critic:21,crjvcbcdpj:11,crucial:1,cseg:10,csv:[1,2,5,6,11,12,16],cuoqjweiq3:11,current:1,custom:1,d8ivcgdgok:11,da:16,dai:[1,5,7,10,11,15,16],dann:16,data:[1,2,4,5,9,13,14,18,20,21],databas:[2,4,5,6,7,11,14,15,18,20],databasenam:13,datafram:[0,1,4,5,6,7,11,12,15],datatyp:[1,7,9,10,13,16,21],date:[0,1,2,8,9,11,14],datetim:10,datetime64:1,db1:5,db2:[5,7],db:[0,1,4,5,7,9,10,11,12,13,15,16,20,21],db_dir:16,db_minim:9,db_updat:21,dciqmi1j2z:11,de:[1,16],decim:16,def:16,defin:[1,3,4,7,9,10,13,16,18,20,21],definit:[1,2,12],delet:1,delim_whitespac:16,dem:16,demonstr:7,den:16,denn:16,depart:16,depend:6,depth:[1,13],der:16,deriv:1,describ:16,descript:[1,6,11,13,15,16],detect:5,determin:4,deu:[1,9,11,16,21],dev:[2,9,10,11],develop:[1,2,10],df:[1,9,21],df_age:7,df_likabl:7,dfg:16,dggwidgqh:11,dict:[1,3,4,5],dictionari:[1,3,4,5,13,20],die:16,differ:[1,4,5,7,10,13,16],dimens:10,directli:[0,1],directori:[1,4,5,8],discard:[1,7],disgust:16,disk:[1,9,18],displai:7,dlccjcr6ag:11,dlvhyaiblj:11,dmuypjzfwx:11,doc:[6,8],docstr:6,document:[6,10,16],doe:[1,10],doi:16,domin:10,don:7,download:16,draw:[1,4,6],drkmloxqzv:11,drop:1,drop_column:1,drop_fil:[1,6],drop_index:1,drop_tabl:1,dspskratv2:11,dswvbofwxh:11,dtype:[1,5,6,7,9,10,11,13,16,21],dummi:[9,10],duplic:[1,6],durat:[4,6,16],dw73mfmgqk:11,dwrgqo1jjl:11,dyyygam5it:11,dz2nnxsq8z:11,dz8inrmxy1:11,e5qr2kha28:11,e:[1,8,10,13,16,19],each:[5,9,10,12,13,15,16],easi:18,easier:21,easili:14,eben:16,edu:16,eebb85363o:11,efr0cxj0tx:11,ehmijhf9xf:11,ehn0cdjjjg:11,eisschrank:16,either:[1,15],ejrc7tmi8k:11,element:13,elif:1,els:16,emot:[1,9,10,12,16],emotion_map:16,empti:[1,4,15],en:5,encod:[16,20],end:[1,2,5,7,11,15],eng:[1,5,9,11,21],english:[1,5],enough:[10,14],enpx3x9edf:11,ensur:6,entri:[1,5,9,12,16],env:[8,19],environ:[8,19],ep5yilvt95:11,equal:6,erkannt:16,erkennung:16,erklaerung:16,error:[1,5,6,8],es:16,especi:21,etc:[4,14],euq6yyabyh:11,evalu:16,even:[5,8],everi:[1,4,10,13,16],everyon:8,evvlnycryh:11,exampl:[0,1,5,7,9,12,20],execut:8,exist:[1,5,6,16,21],expect:[1,3],expected_typ:3,experi:14,expir:[1,13],expiri:1,explicitli:[4,13],express:16,extend:[1,10,16],extend_index:1,extens:1,extract_arch:16,eymvtwb3vz:11,f1:[1,5,15],f1cytzqxdg:11,f2:[1,5,15],f3:[1,5,15],f4:[1,5],f8i2ey1f5h:11,f:[10,16],fals:[0,1,4,5,10,11,16],far:7,fc4rqq7cjj:11,fdoskwjbkf:11,fdqwugwnuz:11,fear:16,featur:[11,21],feel:8,felt:16,femal:[1,2,7,10,16,20,21],fetqbycz2p:11,few:8,ff1v2fzl7q:11,fgymusrdr2:11,field:[1,2,3,13,20],file:[1,2,4,5,6,7,8,11,12,13,14,15,16,18,20,21],file_dur:4,file_root:4,filenotfounderror:5,filewis:[1,2,4,5,7,9,10,11,13,16,20,21],filewise_index:[5,10,13,15,16],fill:[1,7,9],fill_valu:1,filter:14,find:[8,21],first:[4,7,16],fit:[1,13],fix:6,flac:1,float64:5,fly:10,fmh8zgyrbt:11,fn7hzx38hc:11,fncbcwxeiw:11,folder:[1,5,9,12,16,18],follow:[0,1,7,8,9,10,12,14,15,16,20,21],foo:9,forbid:6,format:[1,2,4,6,9,11,13,14,16,18],found:[1,5],fqgrioayzf:11,fqntmm86ve:11,frame:[1,5,13],free:8,french:21,fresh:21,from:[1,4,5,6,7,8,9,15,16,21],from_i:16,frqkeztfrr:11,func:1,fund:16,further:[5,14,16],g2qeodphkj:11,g:[1,8,10,13,19],gcghn05obb:11,gefahren:16,gehen:16,gender:[1,7,10,14,16,20,21],gender_and_ag:7,gener:[4,6,8,14],gerad:16,german:16,get:[0,1,4,7,8,10,11,15,20],getragen:16,gg1nmthrn5:11,git:8,github:[8,9,11],gitlab:13,given:[1,5],gmbh:11,goflfjl6ci:11,gold:[9,11,16],gold_id:10,gold_standard:10,goodby:20,gpi01yyeb:11,ground:2,grow:21,gt:[7,11],gty6o2luo1:11,gzafp2iu9g:11,h5qn1m0ejx:11,h:11,ha2bfspjzg:11,ha:[1,3,5,9,10,15],habe:16,haben:16,had:16,handl:14,happen:7,happi:16,haus:16,have:[0,1,4,5,7,9,10,13,16,18,20,21],hcn49ecdsz:11,header:[1,10,11,12,18],header_onli:1,height:13,hello:20,here:[10,20],heut:16,hf4m1duesl:11,hffloiqrhh:11,highlight:[7,12],hinlegen:16,hk2dron4yt:11,hnucgwnjsm:11,hoch:16,hochgetragen:16,hold:[1,10,12,20],home:[8,19],host:13,how:[0,9],howev:5,hqtdqakof6:11,html:[6,8],http:[1,8,9,11,13,16],htx5pcq6gz:11,human:[1,2,9,10,11,13,14,16],hundr:21,hut3rqxuci:11,hwdmjwtfo3:11,hz:[1,4,13],i1:5,i2:5,i3:5,i4:5,iccvyrdpd7:11,ich:16,id:[1,4,5,9,10,12,13,16,20],ident:6,identifi:[1,3,5,13],idx:5,ieyhybwo45:11,ihm:16,iloc:0,immer:16,implement:[6,12,18],improv:8,includ:[1,20],inconsist:8,increas:10,indent:1,index:[1,2,5,6,7,9,10,12,15,16],index_col:16,index_ex:1,index_typ:[4,7,21],indexfield:1,indextyp:[1,4,7,20,21],indic:[1,5,6],individu:14,info:16,inform:[0,1,7,9,12,13,14,16,20,21],initi:6,inplac:1,input:[4,5,6],inqee4jah8:11,insid:14,instal:18,instanc:[12,21],instead:[6,8,10],instruct:18,int64:1,integ:[1,2,4,7,9,21],intend:1,interest:[4,20],intern:[9,11,21],interpret:6,intersect:6,invalid:[1,4],invalid_id:3,invalid_valu:3,invit:8,involv:1,io:[5,9],ipwr95vbpw:11,is_filewis:1,is_numb:16,is_numer:1,is_port:[1,6],is_seg:1,isjph1hmpj:11,iso:5,issu:8,ist:16,iter:1,its:[1,5,16],j2prlnjhzd:11,j7s8nlmpzy:11,j:11,jcdgsyv4bp:11,jcrorbzsxv:11,jetzt:16,jln8vi9oqh:11,jmlflmg9kp:11,job:[1,5],join:[5,16],join_label:6,join_schem:6,jtnnjqc299:11,jueyj1zbjg:11,juqztnl6in:11,just:[7,9],jvf8qggc7m:11,jwsz0jxvfr:11,k0oo2hxgph:11,k63z8tzvrz:11,karl:16,kbsqtml8xp:11,keep:[1,6,7,20],kei:[1,4,13,16,20],kept:[1,5],keyword:5,kgtaqcvf3f:11,khjsvxqacl:11,kjp8zfmrjb:11,klktth24w0:11,kwarg:5,kxhurazsoi:11,kxmxlbvw07:11,ky0vrqys9t:11,l0vlogfwef:11,l4xptdygnk:11,l:16,label1:11,label2:11,label3:11,label:[0,1,5,7,9,10,11,13,14,15,16,21],label_map_int:[9,11],label_map_str:[9,11],lablaut:16,labsilb:16,lafzvazhdt:11,lambda:[10,16],languag:[1,5,9,10,11,13,16,21],lappen:16,last:[1,5],later:[1,10,21],latest:8,latin:16,ldzdvjcitg:11,learn:[14,18],least:[1,5,6],lejuavqtuk:11,len:[1,21],length:1,let:9,level:[1,5,10],librispeech:10,libsvm:11,licens:[1,6,11],license_url:[1,6],liegt:16,likabl:7,like:[10,16],likeabl:7,likewis:20,limit:1,link:[1,8,12,13,14,15],linkcheck:8,list:[1,3,4,5,10,12,13,16,20],listdir:16,ll:8,load:[1,6,16,18],load_data:1,load_header_from_yaml:1,loc:16,locat:[1,5,14,16],longer:[6,7],look:[0,9,16,18],lower:[1,10],lowercas:10,lt:[7,11],ltuj7biar1:11,lyvvrf5vsq:11,m2flp179di:11,m8dowto8dc:11,m8l0nsyl6t:11,m9grki6hsz:11,m:[8,10],machin:[1,2,5,11,14,18],made:[8,11],mai:14,main:14,maintain:1,make:[6,8,14,16,18],male:[1,2,7,10,16,20,21],mandatori:13,mani:14,map:[1,5,10,16],map_fil:[0,1,6],map_languag:[16,21],mark:13,match:[1,5],maximum:[1,7,9,10,11,13,16,21],mbrldbixon:11,mean:[7,10,15],media:[2,4,5,7,9,11,12,15,16,18],media_id:[1,4,9,11,13],mediaid:13,mediatyp:[1,9],mention:6,merg:[14,21],meta:[1,7,10,12,13,14,16],metadata:[10,14],method:0,metric:10,mfa:10,microphon:[9,11,16],might:[5,10,21],min:1,minim:[4,7,9,15,20,21],minimum:[1,7,9,10,11,13,16,21],miss:[1,5,6],mit:16,mittwoch:16,mkdir:16,modif:1,more:[0,1,5,9,15,21],move:1,mp4:[1,13],ms:1,mstzlijc63:11,mtjvib1fpn:11,multi:[1,14],multiindex:[1,5],multipl:[10,12,20],multipli:[1,5],must:[1,5,10,13],mvuyrcuzhu:11,mwcszxfwlc:11,mydata:10,mydb:1,n49lvawkwm:11,n:[1,13,16],na:[7,11],nach:16,name:[1,3,4,5,6,9,11,13,16,20,21],nan:[1,5,7,11],nat:[0,1,5,6,7,11,15],nbixmbzihn:11,nc:2,ndarrai:1,ndcmn4k9fv:11,ndx7nrwkkn:11,neben:16,necessari:8,need:[5,8],neg:1,neutral:[1,16],newer:1,newest:8,next:7,ngr6omyv66:11,nlhcxypv5h:11,nlkejzb7kj:11,nnnxcokjxx:11,no_schem:[9,11],non:1,none:[0,1,4,5,11,13,16],notabl:6,notat:13,notconformtounifiedformat:6,note:[6,13,16,20,21],now:[7,16,21],np:[5,10],ns:1,nth:13,nullabl:6,num_fil:[4,7,21],num_segments_per_fil:4,num_work:[1,5,6,10,16],number:[1,4,5,13,15],numer:[1,13,14],numpi:10,nuy6vqgj5o:11,nxftjgquoq:11,nyo5pljymc:11,o8fsaroj82:11,oben:16,obj1:5,obj2:5,obj:[1,5],object:[1,4,5,6,9,13,16,18],occurr:5,offer:10,offici:10,often:16,ohpej39lii:11,oinhnpeq95:11,okiaxmfe5u:11,older:6,omiss:8,omit:13,onc:20,one:[1,4,5,6,9,10,13,15,18],ones:10,onli:[1,4,5,9,10,13,21],op3renhgxf:11,oper:[1,7],option:[1,4,5,13],oqbhdqkquo:11,orankjox9c:11,order:[1,4,5],ore:1,organ:[1,6,11,13],origin:[1,5,13,20,21],os:[0,16],other:[1,2,6,7,8,10,11,13,16],otherwis:[1,4],our:21,ourselv:16,out:21,output:[1,5,6],output_fold:5,ovcukmwnvx:11,over:21,overlap:[1,5],overwrit:[1,5,6,21],overwritten:1,ovgriwimq7:11,oxmofxkc8t:11,oxwhdtgc2n:11,oy7klhzwxt:11,ozmrozeds:11,p35wsppvhb:11,p40zo4yffv:11,p60jvt7nuc:11,p9fjq8wkqi:11,p_none:[1,4],packag:8,page:8,panda:[0,1,4,5,6,7,10,12,15,16,20],paper:16,papier:16,parallel:[1,5],param:[10,16],paramet:[1,3,4,5],pars:14,parse_nam:16,part:[1,13,15,16],particip:16,pass:[1,4,20],past:16,path:[0,1,5,15,16],pd:[1,5,10,11,15,16],pdf:16,per:[1,4,9,10,13],permiss:[1,2],phn0egsvnw:11,pho:10,phonem:10,phonet:10,pick:1,pick_column:1,pick_fil:[1,6],pick_index:1,pick_tabl:1,pickl:[2,6],pip:[8,19],pixel:[1,13],pkl:[1,2],place:[1,5,16],plai:9,platz:16,pleas:8,po5utd3mu7:11,point:18,portabl:[1,4],posit:[1,5,16],possibl:[1,4,5,8,9,12,14,20,21],postfix:10,ppfbmytaez:11,pprvr6mlcq:11,pre:[3,18],predict:11,prefix:1,prepar:16,present:[1,5,16],preserv:5,previou:5,principl:10,probabl:[1,4],processor:[1,5],produc:16,progress:[1,5],progress_bar:[5,10],project:[6,8,16,18],prop1:11,prop2:11,properti:1,provid:[1,4,5,9,10,16],psoj2ooszg:11,psu:16,pull:8,push:8,pypi:8,pyrxr7fe7:11,pytest:8,python3:[8,19],python:[8,19],q2l1mry9kx:11,q7a7zk1w6p:11,qfz2l1aiaz:11,qhmqi9l2dp:11,qjaokvrutk:11,qowlmfjw5g:11,qtjbdvjzni:11,qwdepcyklc:11,qzk0eijg:11,r1:9,r2:9,r2oxncmip7:11,r3:9,r:8,rais:[1,3,4,5,6],random:4,randomli:1,rang:[1,4,9,21],rate:[1,4,11,12,13,16],rater1:10,rater2:10,rater:[2,9,11,12,14,16],rater_id:[1,4,9,10,11,13,16],raterid:13,ratertyp:[1,10,13],raxmbav1v3:11,rc4st33zhk:11,rckbntajor:11,rdoyddkhrf:11,re:8,read:[5,16],read_csv:[9,16],readabl:10,real:[9,16],recommend:[2,10],reconstruct:5,record:[14,16,21],redund:10,redundantargumenterror:6,refer:[1,5,15,18],referenc:[1,5,7,12],regex:16,rel:[5,15],relabel:21,releas:6,relev:5,reli:1,rememb:16,remov:[1,5,6],rep1:16,rep:16,repetit:16,replac:[1,5,16,18],replace_label:[1,6],repositori:8,repres:[1,13,18],request:[8,16,20],requir:[1,5,8,9],research:[2,16],resolut:[1,13],respect:1,restrict:[1,2],result:[0,1,6,7,9],rgb:[1,13],right:5,rjnz7rwuo5:11,role:10,root:[1,4,5,6],round_trip:6,row:[1,11,12,21],rst:8,rule:1,run:[14,19],run_task:[10,16],runter:16,runtimeerror:[1,4],rwoi84wh7h:11,s1jwggx4ja:11,s86zfu3akp:11,s8fdsb6qic:11,s9doujd1rg:11,s9rjmat8ru:11,s:[1,5,9,10,12,16],sa:2,sad:16,sagen:16,same:[1,5,10,20],sampl:[1,4,13],sample_gener:4,sampling_r:[1,4,10,11,13,16],satz:16,save:[1,4,5,6,16,18],scalar:1,scheme:[2,4,5,6,7,9,11,12,16,21],scheme_id:[1,4,5,9,10,11,13,16,21],schemeid:13,schwarz:16,se462:16,search:14,second:[1,6,7,13,21],see:[1,5,10,13,16],segment:[1,2,4,5,7,9,10,11,12,14],segmented_index:[5,6,15],sein:16,select:1,self:1,semant:6,sentenc:16,separ:[5,10],sequenc:[1,3,4,5],seri:[0,1,4,5,6,15,20],set:[1,4,5,6,9,10,13,15,16],set_level:1,sever:[5,10,18,20],should:[1,4,5,8,10],show:[1,5,9,12],shutil:16,sich:16,sie:16,sieben:16,silb:16,simpl:14,simpli:[7,8],sind:16,singl:[1,5,10],situat:16,six:16,size:1,sj1rolsuno:11,sketch:12,skrrydoaig:11,slow:1,small:[4,16],so:7,soavbysfax:11,some:[6,7,10,21],sort:16,sourc:[1,2,3,4,5,8,9,10,11,13,14,19,21],soweit:16,spawn:5,speaker1:10,speaker2:10,speaker:[1,12,14,16,20],speaker_map:16,special:10,specif:[0,1,2,4,5,9,12,15,18],specifi:[1,14],speech:16,speed:1,sphinx:[6,8],spk1:20,spk2:20,spk3:20,split:[2,4,9,11],split_id:[1,4,9,10,11,13],splitid:13,splittyp:[1,9,10,13],spoken:16,sprji4txp:11,sqb4sawlcz:11,sqnfqf6le2:11,squeez:16,src:16,src_dir:16,ss09lifqbw:11,ssag7f7ybz:11,ssq00xw0fy:11,st4bepwodt:11,stai:8,stamp:4,standard:11,start:[1,2,5,7,11,15,18,21],state:16,statist:14,stehen:16,step:8,still:8,storag:[1,2],storage_format:1,store:[1,5,6,7,9,10,13,14,18,21],str:[1,2,3,4,5,7,10,11,16,21],str_len:1,string:[0,1,2,3,5,9,11,20],stringio:[5,9],structur:5,stunden:16,sub:[1,4,9],subset:1,sunasr938k:11,support:[5,6],sure:[6,16],svsafex71r:11,swz2ipekeo:11,sxiymjtwj:11,syfo97trmi:11,syntax:6,t2nkzkqoga:11,t:[7,16],tabl:[0,2,4,5,6,9,11,12,20,21],table_ex:1,table_id:[1,4,7,10,12,21],table_str:1,tableid:13,tablestorageformat:1,tabletyp:[1,13],tag:8,take:[10,11],task_func:[10,16],technic:16,ten:16,test:[0,1,2,6,7,10,11,13,15,20,21],tevg1enlvk:11,text:10,tfarnes4e4:11,tfvcgl4f2r:11,than:1,thei:16,them:[1,14,20,21],thi:[1,5,6,7,8,9,10,13,16,18,20,21],thing:8,those:[8,21],thousand:10,thread:5,three:9,through:1,time:[1,2,4,9,10,11,15,16,21],timedelta64:1,timedelta:[1,4,5,10],timestamp:1,tisch:16,to_i:16,to_pandas_dtyp:1,to_replac:16,to_segmented_index:6,to_timedelta:[1,10,16],togeth:[18,21],took:16,tool:14,total:5,train:[1,2,9,10,11],transcrib:20,transcript:[10,16,20],transcription_map:16,treat:1,trinken:16,truth:2,tt6735ug00:11,tt8migbbu5:11,tt9wssefpr:11,tupl:4,two:[1,7,10,15],twvnqx9psw:11,txrb6gcp4i:11,txt:[8,16],type:[1,2,3,4,5,6,7,9,10,11,13,15,16,21],typic:13,u5y6z2cfpj:11,u6yj2gbb1g:11,u:16,ucrekhhle:11,ufdwekw6bl:11,ufunb1dxm0:11,uk0hw0sio9:11,ulwmssklr6:11,ulyxnxp5df:11,ulzhfiqxun:11,und:16,under:[1,5],underli:1,understood:14,unexpect:3,unifi:14,union:[1,4,6],uniqu:[10,13],unit:[1,4,9,10,11,16],unittest:[4,9,11,21],univers:[14,16],unknown:[1,3],unrestrict:[1,2,9,10,11,16,21],unter:16,until:13,up:[1,4,8],updat:[1,5,6,7,8],update_other_format:[1,6],upward:6,url:[1,13],urllib:16,urlretriev:16,us:[1,2,4,5,6,7,8,10,13,14,16,20,21],usag:[1,9,10,11,13,16,18,21],usecol:16,usual:1,ut0qyy5cr8:11,util:[6,9,16,21],utt01:9,utt02:9,utt03:9,utter:16,uwlva4dhwk:11,uyaqdxhaip:11,uyckgdp3rf:11,v2bxca1ugj:11,v8yo5sg2oh:11,vaaiwwjnzk:11,valenc:10,valid:[1,3,8],valid_valu:3,valu:[1,3,4,5,6,7,13,15,16,20],valueerror:[1,5],values_dict:15,values_list:15,vbhajsg8x1:11,vdcadbobhz:11,verbos:[1,5,6],veri:10,version:[8,10,16],vfuyqjohzl:11,video:[1,2,12,13,18],video_channel:[1,11,13],video_depth:[1,11,13],video_fp:[1,11,13],video_resolut:[1,11,13],view:1,viewdoc:16,virtual:19,virtualenv:[8,19],virut:8,voczmf086q:11,vote:[2,10],vrzdkuknzc:11,vs:10,vwe65ojub2:11,vxmaalpxwi:11,w0pau82fqj:11,w2axi3abyv:11,w2zqkzbkpy:11,w5h0rdhvg3:11,w9wg3mgjrb:11,w:16,wa:[1,4,14,16],wagner:11,wai:[8,14,18],want:[5,7,9,21],wav:[0,1,4,7,9,10,11,13,16,20,21],we:[1,7,13,16,20,21],webcam:11,webpag:13,websit:1,wegbringen:16,well:[1,4,6,16],were:16,wfidbsxepm:11,what:13,when:[3,5,6,10,13,16],where:[1,5,13],which:[0,1,5,6,7,9,10,12,16],who:13,whole:[1,7,14,15],width:13,wieder:16,wierstorf:11,wir:16,wird:16,wis7ibqc1i:11,within:[5,10],without:[1,4,9],wjoewbx7x:11,wmn06iebzo:11,wo:16,wochenenden:16,word:[10,20],work:[7,20],worth:10,would:[1,9,16],write:[9,10],written:[12,18],wrong:[6,21],wspuwhgawo:11,wumq2jxwo0:11,www:1,wybad5ftsm:11,wyk7mzcs31:11,x02xxkjnhq:11,x3f6gohpom:11,x:[8,10,13,16],xa0:16,xbydycygmi:11,xcraf28sfb:11,xcxlear0r0:11,xdn8ymzg8:11,xfbybjg9u6:11,xg3ve4aohb:11,xk8ah6itum:11,xnzshiyqg:11,xok984ccnj:11,xq0ctmgead:11,xxx:10,y:[8,16],yaml:[1,11,12,16,18],ydvt2dq4ed:11,ye:13,yet:[1,4,15],yield:16,yjtsog04qu:11,ykdk6lkq8m:11,yl8q6kctgk:11,ylutkqy9hv:11,you:[0,1,4,5,7,8,9,10,16],your:[1,2,7,8,9],yred9bmmdk:11,yyyjsqke1:11,yzcgr7gloh:11,yzoxnhicq3:11,z750v8skev:11,z:8,zfsqrxrrdz:11,zip:16,zjptkyurwo:11,zuxzcdbumi:11,zxluu5qgnk:11},titles:["Working with a database","audformat","audformat.define","audformat.errors","audformat.testing","audformat.utils","Changelog","Combine tables","Contributing","Create a database","Conventions","Example","Database","Header","Introduction","Tables","Emodb example","Index","audformat","Installation","Map scheme labels","Update a database"],titleterms:{"0":6,"01":6,"02":6,"03":6,"04":6,"05":6,"06":6,"07":6,"1":6,"10":6,"11":6,"12":6,"14":6,"18":6,"2":6,"2020":6,"2021":6,"21":6,"22":6,"23":6,"28":6,"3":6,"31":6,"4":6,"5":6,"6":6,"7":6,"8":6,"9":6,"new":8,access:0,add_tabl:4,annot:[10,16],assert_index:1,audformat:[1,2,3,4,5,12,13,15,16,18],audio:9,badiderror:3,badtypeerror:3,badvalueerror:3,build:8,chang:0,changelog:6,column:[1,13],combin:7,concat:5,confid:10,contribut:8,convent:10,creat:[8,9,16],create_audio_fil:4,create_db:4,csv:9,data:[0,10],databas:[0,1,9,10,12,13,16,21],datatyp:2,defin:2,develop:8,disk:[12,16],document:8,durat:[5,10],emodb:16,error:3,exampl:[11,13,16],exist:9,file:[0,9,10],filewis:15,filewise_index:1,gather:16,gender:2,get:16,gold:10,hard:12,header:[13,16],implement:[13,15],index:17,index_typ:1,indexfield:2,indextyp:2,inform:10,inspect:16,instal:[8,19],intersect:5,introduct:14,join_label:5,join_schem:5,label:20,licens:2,map:20,map_languag:5,media:[1,13],mediatyp:2,metadata:16,minim:13,name:10,part:12,rater:[1,10,13],ratertyp:2,read_csv:5,referenc:0,releas:8,run:8,scheme:[1,10,13,20],segment:15,segmented_index:1,sourc:16,speaker:10,split:[1,10,13],splittyp:2,standard:10,store:[12,16],tabl:[1,7,10,13,15,16],tablestorageformat:2,tempor:10,test:[4,8,9],to_filewise_index:5,to_segmented_index:5,union:5,updat:21,us:9,usag:2,util:5,valu:10,version:6,work:0}})