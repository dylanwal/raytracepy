"""
This python file contains the data related to lenses.
* Relative light intensity emitted at a given angle
* Relative transmittance through a diffuser

"""

from numba import njit
import numpy as np

from .. import dtype


@njit
def plane_func_selector(fun_id, x):
    if fun_id == 1:
        x_cdf, cdf = LED_theta_cdf()
        return sample_a_distribution(x_cdf, cdf, x[0]) / 360 * (2 * np.pi)
    if fun_id == 2:
        x_cdf, cdf = diff_cdf()
        return sample_a_distribution(x_cdf, cdf, x[0]) / 360 * (2 * np.pi)
    if fun_id == 3:
        return diff_trans_fun(x)
    if fun_id == 4:
        return mirror_reflect_fun(x)
    if fun_id == 5:
        return ground_reflect_fun(x)
    if fun_id == 6:
        return pcb_reflect_fun(x)
    else:
        return np.ones_like(x, dtype=dtype) * -1000000


def LED_theta_fun(x):
    """
    Data for Lumileds LUXEON C Color Line. Data obtained from manufactures datasheet.
    https://lumileds.com/products/color-leds/luxeon-c-colors/
    :param x: Angle in degrees
    :return: Interpolated value of relative emitted light intensity [0, 1]
    """
    # angle of emitted light
    x_len = np.asarray([
        -180,
        -124,
        -122.741024035727,
        -119.684258904814,
        -116.627106146493,
        -113.569307342493,
        -110.50944119232,
        -107.86373877591,
        -105.909983584157,
        -104.512845198421,
        -103.53250318259,
        -102.691448615184,
        -100.155730091936,
        -99.0340871186469,
        -98.3034675996358,
        -97.6278287214063,
        -96.7859450620458,
        -96.2248866919852,
        -94.5312887115133,
        -93.6903525858156,
        -92.8488242515787,
        -92.0071774756339,
        -91.1654122579813,
        -90.3235417588106,
        -89.4817633809683,
        -88.6405903718549,
        -87.7992989210337,
        -86.9582443536281,
        -85.9772212979771,
        -84.8564370270692,
        -83.7355639248805,
        -82.6153126416579,
        -81.4952390209971,
        -80.3753430628981,
        -79.2561577550462,
        -78.1367947846325,
        -76.8779311707192,
        -75.4798689396614,
        -73.9423297534456,
        -72.4055249058185,
        -70.8692412017059,
        -69.1937648025341,
        -67.5190582744634,
        -65.7050473777232,
        -63.7518251736555,
        -61.7988567732475,
        -59.7067468615184,
        -57.4757492421278,
        -55.2451069478608,
        -53.0147311474364,
        -50.6452406260772,
        -47.9977447150791,
        -45.072281907997,
        -42.0079903449217,
        -38.9446678503652,
        -35.8821852151916,
        -32.8205101371169,
        -29.7596426161411,
        -26.6995826522643,
        -23.6403302454864,
        -20.5814977684,
        -17.5233436392766,
        -14.4651249055854,
        -11.8242732154831,
        -9.04515698261403,
        11.1343341862208,
        13.877060550283,
        16.9323720784181,
        19.9874574905654,
        23.0420260661694,
        26.0958839915264,
        29.1491604757721,
        32.2016617052028,
        35.2533553775346,
        38.3042737950514,
        41.3544492600372,
        44.4035587496524,
        47.3135293813483,
        50.0842480971312,
        52.7156793644887,
        55.0697971391839,
        57.2850716218581,
        59.4999907794087,
        61.5758636020106,
        63.5133499791789,
        65.4503795097598,
        67.3870537152172,
        69.1850580611543,
        70.8442487254975,
        72.3647323057836,
        73.7462363860847,
        75.1275983363364,
        76.508249636341,
        77.8887588062962,
        79.1305490480236,
        80.2340112191592,
        81.3374733902947,
        82.440402573745,
        83.5429764320717,
        84.6453726278367,
        85.7474134984781,
        86.8493655378387,
        87.8132737667062,
        88.6391677955076,
        89.4651802660168,
        90.2910742948182,
        91.1160207899568,
        91.9416779353426,
        92.7668613138969,
        93.5926369009905,
        94.4187678132076,
        96.6102947334588,
        97.4354781120132,
        98.2592994109273,
        99.0372508684311,
        100.184391709381,
        102.78782927909,
        103.889337162046,
        105.13160709269,
        106.651901166244,
        109.00586550509,
        112.057559177422,
        115.11199854389,
        118.167826908568,
        121.223784482383,
        123.446698455213,
        124,
        180
    ])
    # normalized intensity
    y_len = np.asarray([
        0,
        0,
        0.00826858020970777,
        0.00881849429180281,
        0.0101932794970407,
        0.0129428499075165,
        0.0200917329747538,
        0.0328428657533352,
        0.051368096393916,
        0.0680029973772948,
        0.0846378983606734,
        0.100516667481171,
        0.174869633997788,
        0.196545414067039,
        0.213936446913298,
        0.232587699531026,
        0.250230776331579,
        0.261572754274792,
        0.317778556082268,
        0.333405281248472,
        0.350292226186144,
        0.36743121507811,
        0.38482224792437,
        0.402437319841113,
        0.419856357571183,
        0.435987170645974,
        0.452370027675059,
        0.468248796795557,
        0.486332950516124,
        0.506181411916746,
        0.526218906283089,
        0.54493316988939,
        0.56326936756425,
        0.58122749930767,
        0.597673367325329,
        0.614497301274428,
        0.632531046204136,
        0.651131890031005,
        0.670892136047624,
        0.689089709547623,
        0.706178289648731,
        0.723821366449284,
        0.739826157546928,
        0.756623086786639,
        0.774014119632898,
        0.790865058291386,
        0.808161574654785,
        0.825363574535324,
        0.841809442552983,
        0.857688211673481,
        0.873955548556848,
        0.890523237819082,
        0.907309365175036,
        0.923875526898153,
        0.938379510813413,
        0.951096273961864,
        0.962094555603767,
        0.971374355739123,
        0.978935674367931,
        0.984778511490192,
        0.989727738229049,
        0.993233440502406,
        0.996876621296286,
        0.999305408492206,
        1.00031358430938,
        0.99946293596364,
        0.996945360556548,
        0.994402007926858,
        0.991377480475334,
        0.987253124859621,
        0.981616505518145,
        0.974742579491956,
        0.966218911219481,
        0.955976761440458,
        0.94408486941515,
        0.930611974403819,
        0.914870683803845,
        0.898689461938195,
        0.881827721395952,
        0.864209848990828,
        0.84776398097317,
        0.831507145921231,
        0.814494179006412,
        0.797238169707096,
        0.781143362911489,
        0.764076386577893,
        0.746253278381416,
        0.728988267512303,
        0.711975300597484,
        0.695441217195823,
        0.678806316212444,
        0.661868962483913,
        0.64341934502962,
        0.624667274830175,
        0.606368883748459,
        0.58935591683364,
        0.57234294991882,
        0.55419578520968,
        0.535292488637659,
        0.516011126134197,
        0.495973631767855,
        0.475747104435792,
        0.457410906760931,
        0.441028049731846,
        0.424897236657055,
        0.40851437962797,
        0.390115170964536,
        0.373228226026864,
        0.355333105272017,
        0.338698204288638,
        0.32281943516814,
        0.256027787280332,
        0.238132666525485,
        0.217339040296262,
        0.197483577674687,
        0.174869633997788,
        0.0976811729953681,
        0.0765094808347042,
        0.0592318677678769,
        0.0422945140393458,
        0.0255221345354434,
        0.0152799847564211,
        0.0108806720996597,
        0.00943714763415992,
        0.00826858020970777,
        0.00826858020970777,
        0,
        0

    ])
    return np.interp(x, x_len, y_len)


@njit
def LED_theta_cdf():
    """
    cumulative distribution function of LED_theta_fun
    :return x, and cdf
    """
    x = np.array([
        -90.0,
        -88.99441340782123,
        -87.98882681564245,
        -86.98324022346368,
        -85.97765363128491,
        -84.97206703910615,
        -83.96648044692738,
        -82.9608938547486,
        -81.95530726256983,
        -80.94972067039106,
        -79.94413407821229,
        -78.93854748603351,
        -77.93296089385476,
        -76.92737430167598,
        -75.92178770949721,
        -74.91620111731844,
        -73.91061452513966,
        -72.90502793296089,
        -71.89944134078212,
        -70.89385474860336,
        -69.88826815642457,
        -68.88268156424581,
        -67.87709497206704,
        -66.87150837988827,
        -65.8659217877095,
        -64.86033519553072,
        -63.85474860335195,
        -62.849162011173185,
        -61.84357541899441,
        -60.83798882681564,
        -59.832402234636874,
        -58.8268156424581,
        -57.82122905027933,
        -56.815642458100555,
        -55.81005586592178,
        -54.80446927374302,
        -53.798882681564244,
        -52.79329608938547,
        -51.787709497206706,
        -50.78212290502793,
        -49.77653631284916,
        -48.77094972067039,
        -47.765363128491614,
        -46.75977653631285,
        -45.754189944134076,
        -44.7486033519553,
        -43.74301675977654,
        -42.737430167597765,
        -41.73184357541899,
        -40.72625698324022,
        -39.72067039106145,
        -38.71508379888268,
        -37.70949720670391,
        -36.703910614525135,
        -35.69832402234637,
        -34.6927374301676,
        -33.687150837988824,
        -32.68156424581005,
        -31.67597765363128,
        -30.670391061452513,
        -29.66480446927374,
        -28.659217877094967,
        -27.6536312849162,
        -26.64804469273743,
        -25.642458100558656,
        -24.636871508379883,
        -23.63128491620111,
        -22.625698324022338,
        -21.620111731843565,
        -20.614525139664806,
        -19.608938547486034,
        -18.60335195530726,
        -17.597765363128488,
        -16.592178770949715,
        -15.586592178770942,
        -14.58100558659217,
        -13.575418994413411,
        -12.569832402234638,
        -11.564245810055866,
        -10.558659217877093,
        -9.55307262569832,
        -8.547486033519547,
        -7.5418994413407745,
        -6.536312849162002,
        -5.530726256983229,
        -4.52513966480447,
        -3.5195530726256976,
        -2.513966480446925,
        -1.508379888268152,
        -0.5027932960893793,
        0.5027932960893935,
        1.5083798882681663,
        2.513966480446925,
        3.5195530726256976,
        4.52513966480447,
        5.530726256983243,
        6.536312849162016,
        7.541899441340789,
        8.547486033519561,
        9.553072625698334,
        10.558659217877107,
        11.564245810055866,
        12.569832402234638,
        13.575418994413411,
        14.581005586592184,
        15.586592178770957,
        16.59217877094973,
        17.597765363128502,
        18.60335195530726,
        19.608938547486034,
        20.614525139664806,
        21.62011173184358,
        22.625698324022352,
        23.631284916201125,
        24.636871508379897,
        25.64245810055867,
        26.648044692737443,
        27.6536312849162,
        28.659217877094974,
        29.664804469273747,
        30.67039106145252,
        31.675977653631293,
        32.681564245810065,
        33.68715083798884,
        34.6927374301676,
        35.69832402234637,
        36.70391061452514,
        37.709497206703915,
        38.71508379888269,
        39.72067039106145,
        40.72625698324023,
        41.73184357541899,
        42.73743016759778,
        43.74301675977654,
        44.748603351955325,
        45.75418994413408,
        46.75977653631287,
        47.76536312849163,
        48.77094972067039,
        49.776536312849174,
        50.78212290502793,
        51.78770949720672,
        52.79329608938548,
        53.798882681564265,
        54.804469273743024,
        55.81005586592178,
        56.81564245810057,
        57.82122905027933,
        58.826815642458115,
        59.832402234636874,
        60.83798882681566,
        61.84357541899442,
        62.84916201117318,
        63.854748603351965,
        64.86033519553072,
        65.86592178770951,
        66.87150837988827,
        67.87709497206706,
        68.88268156424581,
        69.8882681564246,
        70.89385474860336,
        71.89944134078212,
        72.9050279329609,
        73.91061452513966,
        74.91620111731845,
        75.92178770949721,
        76.927374301676,
        77.93296089385476,
        78.93854748603354,
        79.9441340782123,
        80.94972067039106,
        81.95530726256985,
        82.9608938547486,
        83.96648044692739,
        84.97206703910615,
        85.97765363128494,
        86.9832402234637,
        87.98882681564245,
        88.99441340782124
    ], dtype=dtype)
    cdf = np.array([
        0.0026941213463242374,
        0.005520400531070366,
        0.00847493636787411,
        0.011555229313686093,
        0.014757660764117601,
        0.018077361747784803,
        0.02151531030974387,
        0.025065658951602655,
        0.028725171215006517,
        0.0324918816498327,
        0.036360967518252706,
        0.04032805819162957,
        0.04439372760655156,
        0.04855425633550104,
        0.05280321798280143,
        0.057138599311118296,
        0.06155887139173584,
        0.06605755275820761,
        0.07063225114341257,
        0.07528060538656368,
        0.0799987840122178,
        0.08478469687925595,
        0.08963389230557438,
        0.09454510291359351,
        0.09951762795780011,
        0.10454948841020882,
        0.1096403073673415,
        0.1147884478908362,
        0.11999372333376035,
        0.12525385034873676,
        0.13056872266913136,
        0.1359351125901899,
        0.1413525591912899,
        0.14681959436236763,
        0.1523354497392805,
        0.15789939011716433,
        0.1635104728775891,
        0.16916832766218193,
        0.17487164294832822,
        0.18062041873602797,
        0.1864111801229577,
        0.19224337956659712,
        0.19811622145390104,
        0.20402705853558437,
        0.20997589081164705,
        0.21596201120191402,
        0.221983930069075,
        0.22804164741313,
        0.23413394221657974,
        0.24025758916996384,
        0.2464125882732823,
        0.2525980592235377,
        0.25881102656421123,
        0.2650514902953029,
        0.2713187721808405,
        0.277609841000824,
        0.2839246967552532,
        0.29026282662341624,
        0.2966210320127008,
        0.30299931292310694,
        0.3093973191417905,
        0.3158116875093698,
        0.32224241802584486,
        0.328689320274332,
        0.33514886933903615,
        0.3416210652199573,
        0.348105890531875,
        0.35460142991173044,
        0.36110768335952353,
        0.3676246508752543,
        0.3741493117856009,
        0.3806815635107952,
        0.3872214060508372,
        0.3937691148997212,
        0.400324712075823,
        0.4068881975791424,
        0.4134579803272567,
        0.4200338530879644,
        0.4266148622465157,
        0.4331982735671243,
        0.43978408704979044,
        0.4463709757047591,
        0.45295758522582447,
        0.4595439156129866,
        0.46612996686624547,
        0.47271573898560115,
        0.47930123197105357,
        0.4858864458226027,
        0.4924713805402487,
        0.49905603612399135,
        0.5056404125738309,
        0.5122245098897671,
        0.5188083280718,
        0.5253918671199298,
        0.5319751270341563,
        0.5385581078144795,
        0.5451408094608996,
        0.5517232319734163,
        0.55830537535203,
        0.5648872395967401,
        0.5714688247075471,
        0.5780476514624199,
        0.5846204000486688,
        0.5911870704662937,
        0.5977480589229165,
        0.6043035351941265,
        0.6108534992799235,
        0.6173972608116439,
        0.6239344668269052,
        0.6304651173257072,
        0.636987724850384,
        0.6435013915218879,
        0.6500061173402188,
        0.6564999795963796,
        0.6629816198430674,
        0.6694510380802822,
        0.6759067595984182,
        0.682347573388587,
        0.6887734794507888,
        0.6951826407277499,
        0.7015733117271814,
        0.7079454924490832,
        0.7142974010081556,
        0.7206270855316653,
        0.7269345460196123,
        0.733218195560922,
        0.7394760347993721,
        0.745708063734962,
        0.7519128776246142,
        0.7580884426612461,
        0.7642347588448578,
        0.7703499735188702,
        0.7764310028409811,
        0.7824778468111904,
        0.7884896010695707,
        0.7944645343393484,
        0.8004026466205232,
        0.8063023756429158,
        0.8121618067386599,
        0.8179809399077554,
        0.8237569743192591,
        0.8294886750114308,
        0.8351758933384521,
        0.840816852120976,
        0.8464115513590025,
        0.8519582726720567,
        0.8574564001481513,
        0.8629047243583313,
        0.8683021863949084,
        0.8736474037940941,
        0.8789375768922916,
        0.884172715556267,
        0.8893528469810764,
        0.8944768384720715,
        0.8995424863819506,
        0.9045487179473874,
        0.9094940099108588,
        0.9143770735353062,
        0.9191965530116629,
        0.9239494317014328,
        0.9286342097934455,
        0.9332469814806983,
        0.9377835950455667,
        0.9422402367776819,
        0.946615686964656,
        0.9509041843654162,
        0.9551035869917053,
        0.959212708125203,
        0.9632242545191956,
        0.9671345707245919,
        0.9709427940756482,
        0.9746447105061117,
        0.9782353066128912,
        0.9817114107144277,
        0.9850702095380053,
        0.9883083482861321,
        0.9914243551224132,
        0.994413457288325,
        0.9972719274876661
    ], dtype=dtype)
    return x, cdf


def diff_fun(x):
    """
    Data for ThorLabs DG10-220-MD, Ground Glass Diffusers. Data obtained from manufactures website.
    https://www.thorlabs.com/NewGroupPage9_PF.cfm?Guide=10&Category_ID=220&ObjectGroup_ID=4780
    :param x: Angle in degrees
    :return: Interpolated value of relative intensity [0, 1]
    """
    x_len = np.asarray([
        -180,
        -60,
        -29.0356990035088,
        -26.7167971255572,
        -24.397658388643,
        -22.0782827927664,
        -19.7581176670148,
        -17.4370314230759,
        -15.1137871308122,
        -13.1055553895163,
        -11.6231878282693,
        -10.4583979128282,
        -9.50370365161296,
        -8.65439978575078,
        -7.91090608195852,
        -7.27237818189773,
        -6.52763000285978,
        -5.78312306906138,
        -5.14420917661729,
        -4.39926800138769,
        -3.33872980376897,
        -1.54111063945313,
        0.778738674348446,
        2.77563079281586,
        4.03370059258426,
        4.76498728725491,
        5.39068094057124,
        6.01656759007921,
        6.74727529617493,
        7.58224919480739,
        8.41780208201478,
        9.25436819922833,
        10.3010830446204,
        11.5594182141523,
        12.9224262467676,
        14.8134743661705,
        17.1265337230478,
        19.4418037635749,
        21.7582317812518,
        24.0753703758162,
        26.3927195116805,
        28.7103055065074,
        30.0800082511732,
        60,
        180
    ], dtype=dtype)
    y_len = np.asarray([
        0,
        0,
        0.0110009303169384,
        0.0172759383116831,
        0.0258642688399304,
        0.0367659219016793,
        0.0553786500751032,
        0.0829876325454799,
        0.131673553654433,
        0.191236893644252,
        0.260693408489853,
        0.315974915964613,
        0.377655533833717,
        0.439189957087648,
        0.496478464125367,
        0.557767621385749,
        0.627308169989795,
        0.694492556754163,
        0.75955157295803,
        0.830977051033818,
        0.897233447776689,
        0.958147285928013,
        0.973675584056767,
        0.922488607755165,
        0.859636985493249,
        0.79770370344325,
        0.733645050832748,
        0.671471327693989,
        0.603883257228762,
        0.525461667205805,
        0.452694865598077,
        0.389823943716997,
        0.321072267046304,
        0.260812422808034,
        0.193690874684637,
        0.13792750751918,
        0.0871405596875352,
        0.0579446221685784,
        0.0400582614800768,
        0.0291118683920825,
        0.0202217620005344,
        0.0136449781424887,
        0.0119084510453832,
        0,
        0

    ], dtype=dtype)
    return np.interp(x, x_len, y_len)


@njit
def diff_cdf():
    """
    cumulative distribution function of diff_fun
    :return x, and cdf
    """
    x = np.array([
        -90.0,
        -88.99441340782123,
        -87.98882681564245,
        -86.98324022346368,
        -85.97765363128491,
        -84.97206703910615,
        -83.96648044692738,
        -82.9608938547486,
        -81.95530726256983,
        -80.94972067039106,
        -79.94413407821229,
        -78.93854748603351,
        -77.93296089385476,
        -76.92737430167598,
        -75.92178770949721,
        -74.91620111731844,
        -73.91061452513966,
        -72.90502793296089,
        -71.89944134078212,
        -70.89385474860336,
        -69.88826815642457,
        -68.88268156424581,
        -67.87709497206704,
        -66.87150837988827,
        -65.8659217877095,
        -64.86033519553072,
        -63.85474860335195,
        -62.849162011173185,
        -61.84357541899441,
        -60.83798882681564,
        -59.832402234636874,
        -58.8268156424581,
        -57.82122905027933,
        -56.815642458100555,
        -55.81005586592178,
        -54.80446927374302,
        -53.798882681564244,
        -52.79329608938547,
        -51.787709497206706,
        -50.78212290502793,
        -49.77653631284916,
        -48.77094972067039,
        -47.765363128491614,
        -46.75977653631285,
        -45.754189944134076,
        -44.7486033519553,
        -43.74301675977654,
        -42.737430167597765,
        -41.73184357541899,
        -40.72625698324022,
        -39.72067039106145,
        -38.71508379888268,
        -37.70949720670391,
        -36.703910614525135,
        -35.69832402234637,
        -34.6927374301676,
        -33.687150837988824,
        -32.68156424581005,
        -31.67597765363128,
        -30.670391061452513,
        -29.66480446927374,
        -28.659217877094967,
        -27.6536312849162,
        -26.64804469273743,
        -25.642458100558656,
        -24.636871508379883,
        -23.63128491620111,
        -22.625698324022338,
        -21.620111731843565,
        -20.614525139664806,
        -19.608938547486034,
        -18.60335195530726,
        -17.597765363128488,
        -16.592178770949715,
        -15.586592178770942,
        -14.58100558659217,
        -13.575418994413411,
        -12.569832402234638,
        -11.564245810055866,
        -10.558659217877093,
        -9.55307262569832,
        -8.547486033519547,
        -7.5418994413407745,
        -6.536312849162002,
        -5.530726256983229,
        -4.52513966480447,
        -3.5195530726256976,
        -2.513966480446925,
        -1.508379888268152,
        -0.5027932960893793,
        0.5027932960893935,
        1.5083798882681663,
        2.513966480446925,
        3.5195530726256976,
        4.52513966480447,
        5.530726256983243,
        6.536312849162016,
        7.541899441340789,
        8.547486033519561,
        9.553072625698334,
        10.558659217877107,
        11.564245810055866,
        12.569832402234638,
        13.575418994413411,
        14.581005586592184,
        15.586592178770957,
        16.59217877094973,
        17.597765363128502,
        18.60335195530726,
        19.608938547486034,
        20.614525139664806,
        21.62011173184358,
        22.625698324022352,
        23.631284916201125,
        24.636871508379897,
        25.64245810055867,
        26.648044692737443,
        27.6536312849162,
        28.659217877094974,
        29.664804469273747,
        30.67039106145252,
        31.675977653631293,
        32.681564245810065,
        33.68715083798884,
        34.6927374301676,
        35.69832402234637,
        36.70391061452514,
        37.709497206703915,
        38.71508379888269,
        39.72067039106145,
        40.72625698324023,
        41.73184357541899,
        42.73743016759778,
        43.74301675977654,
        44.748603351955325,
        45.75418994413408,
        46.75977653631287,
        47.76536312849163,
        48.77094972067039,
        49.776536312849174,
        50.78212290502793,
        51.78770949720672,
        52.79329608938548,
        53.798882681564265,
        54.804469273743024,
        55.81005586592178,
        56.81564245810057,
        57.82122905027933,
        58.826815642458115
    ], dtype=dtype)
    cdf = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.2313220314800655e-06,
        2.58505762518408e-05,
        6.785776266108221e-05,
        0.00012925288125920423,
        0.00021003593204620697,
        0.00031020691502209026,
        0.0004297658301868542,
        0.0005687126775404988,
        0.0007270474570830239,
        0.0009047701688144297,
        0.001101880812734716,
        0.0013183793888438833,
        0.001554265897141931,
        0.0018095403376288594,
        0.0020842027103046682,
        0.0023782530151693583,
        0.0026916912522229283,
        0.003024517421465379,
        0.0033767315228967107,
        0.003748333556516923,
        0.004139323522326016,
        0.00454970142032399,
        0.004979467250510843,
        0.005428621012886577,
        0.005897162707451192,
        0.0063850923342046876,
        0.0068924098931470645,
        0.0074191153842783205,
        0.007965208807598458,
        0.008530690163107476,
        0.009115559450805376,
        0.009767844519487643,
        0.010567800606578748,
        0.011519148381351717,
        0.012672586414427162,
        0.014028114705805084,
        0.015627198549563702,
        0.01748278077672984,
        0.019677457293377165,
        0.0223099123806818,
        0.02541149805678383,
        0.029162199983903507,
        0.03356201816204084,
        0.03902639254160826,
        0.045634360928685926,
        0.05363757013547304,
        0.06325933848896735,
        0.0749995850446403,
        0.08929870743648005,
        0.10618780519906267,
        0.12650932478454227,
        0.15079035103330626,
        0.17965537449679916,
        0.21365414144871406,
        0.2527376083386514,
        0.297178135753737,
        0.3452561527932354,
        0.39546389661038744,
        0.4474725312160446,
        0.4998464471245716,
        0.5525856443359685,
        0.6044100977746473,
        0.6548357106037989,
        0.7028804356403351,
        0.7472725674361569,
        0.7863310638095128,
        0.8201615554363805,
        0.8488829725827929,
        0.8729209139737251,
        0.8930111507215895,
        0.9097657101368193,
        0.9239065794806346,
        0.9353600822676379,
        0.9448263427488882,
        0.9526834041755884,
        0.9592472532722678,
        0.9646129040807934,
        0.9690193714538162,
        0.972737687213642,
        0.9758121916625664,
        0.9784653232767045,
        0.9806970820560563,
        0.9826485739313793,
        0.9843422661186026,
        0.9858052104528962,
        0.9870588019519327,
        0.9881168765477518,
        0.9890200904962425,
        0.9897684437974048,
        0.9904432584434557,
        0.991076754569363,
        0.9916885308280963,
        0.9922785872196557,
        0.992846923744041,
        0.9933935404012525,
        0.9939184371912899,
        0.9944216141141534,
        0.9949030711698429,
        0.9953628083583586,
        0.9958008256797002,
        0.9962171231338679,
        0.9966117007208617,
        0.9969845584406815,
        0.9973356962933272,
        0.997665114278799,
        0.9979728123970969,
        0.9982587906482208,
        0.9985230490321707,
        0.9987655875489466,
        0.9989864061985486,
        0.9991855049809766,
        0.9993628838962306,
        0.9995185429443108,
        0.9996524821252168,
        0.9997647014389489,
        0.9998552008855071,
        0.9999239804648913,
        0.9999710401771015,
        0.9999963800221378
    ], dtype=dtype)
    return x, cdf


@njit
def diff_trans_fun(x):
    """
    Data for transmittance through a typical ground glass diffuser. Data obtained from:
    Ching-Cherng Sun, Wei-Ting Chien, Ivan Moreno, Chih-To Hsieh, Mo-Cha Lin, Shu-Li Hsiao, and Xuan-Hao Lee,
    "Calculating model of light transmission efficiency of diffusers attached to a lighting cavity,"
    Opt. Express 18, 6137-6148 (2010) DOI: 10.1364/OE.18.006137.
    https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-18-6-6137&id=196561
    :param x: Angle of incidence in degrees
    :return: Interpolated value of relative transmittance [0, 1]
    """
    x_len = np.asarray([
        -180,
        -90,
        -75.946319465293,
        -66.540292003097,
        -57.1821083002828,
        -47.830322309479,
        -38.0513499612874,
        -28.7089519174554,
        -18.9560571997408,
        -9.60677466146286,
        -0.257213961793098,
        9.05785472529903,
        20.2399426738508,
        28.133328487683,
        37.8798254933872,
        47.189609114036,
        56.5027306713859,
        66.2337897198476,
        75.0675001185077,
        90,
        180
    ], dtype=dtype)
    y_len = np.asarray([
        0,
        0,
        0.30869419312023,
        0.427777402942152,
        0.469646101095011,
        0.501189602920031,
        0.537564420418095,
        0.553956818935958,
        0.548245238358588,
        0.575748445968366,
        0.603700575157615,
        0.575986428492423,
        0.54919392323858,
        0.554681583895587,
        0.538644806990377,
        0.50240115031523,
        0.471544552593739,
        0.430592628027873,
        0.310619688087601,
        0,
        0
    ], dtype=dtype)
    return np.interp(x, x_len, y_len)


@njit
def mirror_reflect_fun(x):
    return np.ones_like(x, dtype=dtype) * 0.95


@njit
def pcb_reflect_fun(x):
    return np.ones_like(x, dtype=dtype) * 0.2


@njit
def ground_reflect_fun(x):
    return np.ones_like(x, dtype=dtype) * 0.02


@njit
def sample_a_distribution(x: np.ndarray, cdf: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Given a distribution x and y; return n points random chosen from the distribution
    :param x: value
    :param cdf: population
    :param n: number of random numbers you want
    :return: np.array of random numbers
    """
    rnd = np.random.random((n,))  # generate random numbers between [0,1]
    return np.interp(rnd, cdf, x)
