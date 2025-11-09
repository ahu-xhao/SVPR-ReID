# coding=utf-8
'''
@Time     : 2023/12/26 09:39:54
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import glob
import os.path as osp

from .bases import BaseImageDataset
from termcolor import colored
import logging
import json
from pathlib import Path
import torch
import copy
CROSS_TIME_TEST = [1598, 1599, 1601, 1606, 1608, 1609, 1804, 1806, 1809, 1812, 1813, 1817, 1821, 1822, 1825, 1828, 1829, 1831, 1834, 1835, 1836, 1837, 1838, 1840, 1841, 1842, 1843, 1844, 1846, 1848, 1644, 1647, 1648, 1649, 1652, 1659, 1663, 1665, 1666, 1667, 1877, 1883, 1885, 1888,
                   1890, 1891, 1894, 1901, 1903, 1906, 1909, 1910, 1913, 1914, 1916, 1918, 1919, 1920, 1928, 1929, 1930, 1931, 1936, 1937, 1939, 1940, 1942, 1943, 1944, 1945, 1946, 2062, 2064, 2066, 2067, 2070, 2072, 2073, 2074, 2077, 2085, 2088, 2089, 2091, 2092, 2093, 2094, 2095, 2096, 2098, 2099]

CROSS_TIME_ALL = [1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1611, 1644, 1645, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1658, 1659, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1875, 1877, 1878, 1879, 1880, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889,
                  1890, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099]

CROSS_TIME_TRAIN = list(set(CROSS_TIME_ALL) - set(CROSS_TIME_TEST))


v501_hard = [94, 103, 109, 113, 115, 127, 129, 137, 146, 148, 160, 166, 167, 169, 171, 173, 184, 256, 274, 298, 299, 302, 317, 329, 334, 347, 355, 358, 367, 369, 370, 382, 408, 418, 454, 472, 491, 525, 529, 540, 541, 555, 558, 560, 574, 581, 586, 587, 589, 603, 608, 613, 623, 631, 639, 644, 645, 647, 652, 657, 658, 664, 675, 687, 722, 731, 753, 761, 763, 768, 792, 797, 805, 806, 813, 824, 834, 837, 838, 853, 899, 963, 973, 986, 1000, 1004, 1034, 1071, 1083, 1084, 1092, 1100, 1105, 1106, 1111, 1117, 1138, 1143, 1144, 1157, 1170, 1182, 1191, 1203, 1221, 1225, 1229, 1232, 1237, 1240, 1241, 1267, 1278, 1279, 1281, 1285, 1293, 1295, 1300, 1304, 1307, 1308, 1312, 1319, 1325, 1326, 1327, 1330, 1334, 1350, 1351, 1352, 1354, 1366, 1369, 1375, 1376,
             1377, 1378, 1384, 1388, 1395, 1398, 1404, 1407, 1413, 1414, 1418, 1421, 1433, 1439, 1441, 1444, 1445, 1456, 1457, 1458, 1467, 1474, 1477, 1478, 1480, 1482, 1484, 1485, 1487, 1488, 1490, 1493, 1506, 1510, 1512, 1521, 1523, 1531, 1532, 1533, 1534, 1536, 1547, 1552, 1553, 1557, 1560, 1564, 1567, 1568, 1570, 1572, 1574, 1582, 1592, 1593, 1594, 1612, 1615, 1636, 1643, 1646, 1657, 1688, 1718, 1723, 1724, 1738, 1740, 1744, 1745, 1746, 1751, 1753, 1757, 1761, 1764, 1765, 1770, 1779, 1780, 1785, 1787, 1792, 1856, 1863, 1868, 1872, 1873, 1891, 1948, 1957, 1961, 1963, 1965, 1966, 1967, 1970, 1976, 1981, 1983, 1984, 1988, 1989, 1995, 1996, 1997, 2005, 2013, 2015, 2018, 2020, 2025, 2026, 2031, 2034, 2038, 2050, 2052, 2053, 2054, 2057, 2100, 2103, 2108]
v501_valid = [2, 8, 9, 14, 25, 31, 32, 35, 36, 37, 39, 42, 48, 49, 50, 52, 54, 56, 58, 62, 63, 66, 72, 83, 84, 91, 93, 94, 96, 101, 103, 104, 105, 108, 109, 113, 115, 127, 128, 129, 133, 134, 137, 141, 143, 146, 148, 152, 155, 160, 162, 166, 167, 169, 170, 171, 173, 184, 188, 189, 191, 198, 199, 207, 211, 214, 215, 219, 227, 232, 234, 235, 241, 251, 254, 256, 258, 267, 273, 274, 275, 280, 282, 286, 291, 294, 298, 299, 300, 302, 303, 308, 314, 317, 322, 326, 328, 329, 333, 334, 335, 338, 340, 345, 346, 347, 349, 355, 358, 367, 369, 370, 371, 372, 373, 374, 375, 379, 382, 385, 389, 394, 398, 399, 402, 407, 408, 410, 411, 412, 413, 416, 418, 419, 422, 425, 427, 432, 435, 437, 445, 454, 456, 462, 472, 477, 479, 484, 491, 493, 500, 501, 515, 525, 527, 529, 535, 536, 540, 541, 546, 547, 551, 555, 558, 560, 561, 563, 566, 573, 574, 577, 580, 581, 585, 586, 587, 589, 590, 597, 598, 599, 603, 608, 613, 614, 616, 623, 631, 632, 639, 640, 644, 645, 647, 651, 652, 657, 658, 659, 662, 664, 669, 675, 679, 681, 684, 685, 687, 699, 706, 710, 722, 724, 731, 734, 740, 742, 743, 744, 745, 753, 757, 761, 763, 765, 768, 771, 775, 784, 786, 788, 789, 792, 796, 797, 798, 802, 805, 806, 810, 813, 816, 819, 822, 824, 834, 836, 837, 838, 840, 842, 852, 853, 859, 861, 865, 867, 869, 871, 872, 878, 879, 880, 881, 884, 887, 891, 895, 898, 899, 903, 905, 923, 924, 925, 942, 943, 947, 958, 963, 964, 969, 973, 977, 981, 985, 986, 990, 991, 997, 998, 1000, 1001, 1002, 1004, 1015, 1017, 1021, 1025, 1029, 1034, 1035, 1036, 1037, 1041, 1043, 1050, 1056, 1060, 1061, 1064, 1066,
              1070, 1071, 1072, 1075, 1079, 1081, 1083, 1084, 1088, 1090, 1092, 1097, 1099, 1100, 1102, 1105, 1106, 1107, 1111, 1112, 1115, 1117, 1120, 1125, 1128, 1130, 1132, 1133, 1138, 1143, 1144, 1145, 1148, 1156, 1157, 1163, 1164, 1169, 1170, 1175, 1182, 1188, 1191, 1195, 1196, 1203, 1206, 1212, 1213, 1216, 1219, 1220, 1221, 1223, 1225, 1229, 1232, 1237, 1240, 1241, 1267, 1278, 1279, 1281, 1285, 1291, 1293, 1295, 1297, 1300, 1301, 1304, 1307, 1308, 1312, 1319, 1324, 1325, 1326, 1327, 1330, 1333, 1334, 1338, 1340, 1346, 1347, 1348, 1350, 1351, 1352, 1354, 1356, 1363, 1366, 1367, 1369, 1374, 1375, 1376, 1377, 1378, 1384, 1388, 1395, 1398, 1404, 1407, 1413, 1414, 1418, 1420, 1421, 1424, 1433, 1439, 1441, 1444, 1445, 1452, 1456, 1457, 1458, 1462, 1467, 1474, 1475, 1477, 1478, 1480, 1482, 1484, 1485, 1487, 1488, 1490, 1493, 1506, 1510, 1512, 1521, 1523, 1531, 1532, 1533, 1534, 1536, 1547, 1549, 1552, 1553, 1557, 1560, 1564, 1567, 1568, 1570, 1572, 1574, 1582, 1592, 1593, 1594, 1612, 1615, 1617, 1620, 1627, 1636, 1639, 1640, 1642, 1643, 1646, 1657, 1660, 1668, 1674, 1676, 1680, 1681, 1682, 1688, 1691, 1693, 1694, 1701, 1713, 1718, 1723, 1724, 1726, 1730, 1731, 1738, 1740, 1741, 1742, 1744, 1745, 1746, 1751, 1753, 1757, 1760, 1761, 1764, 1765, 1770, 1775, 1779, 1780, 1785, 1787, 1792, 1856, 1863, 1864, 1867, 1868, 1871, 1872, 1873, 1891, 1948, 1950, 1951, 1957, 1961, 1963, 1965, 1966, 1967, 1970, 1976, 1981, 1983, 1984, 1988, 1989, 1995, 1996, 1997, 2005, 2013, 2015, 2018, 2020, 2025, 2026, 2031, 2034, 2038, 2050, 2052, 2053, 2054, 2057, 2100, 2103, 2104, 2108]
# v501_easy = list(set(v501_valid) - set(v501_hard))
v501_easy = [2, 8, 9, 14, 25, 31, 32, 35, 36, 37, 39, 42, 48, 49, 50, 52, 54, 56, 58, 62, 63, 66, 72, 83, 84, 91, 93, 94, 96, 101, 104, 105, 108, 128, 129, 133, 134, 141, 143, 152, 155, 160, 162, 166, 167, 169, 170, 184, 188, 189, 191, 198, 199, 207, 211, 214, 215, 219, 227, 232, 234, 235, 241, 251, 254, 256, 258, 267, 273, 274, 275, 280, 282, 286, 291, 294, 300, 302, 303, 308, 314, 322, 326, 328, 329, 333, 335, 338, 340, 345, 346, 347, 349, 355, 358, 369, 371, 372, 373, 374, 375, 379, 382, 385, 389, 394, 398, 399, 402, 407, 408, 410, 411, 412, 413, 416, 419, 422, 425, 427, 432, 435, 437, 445, 456, 462, 477, 479, 484, 493, 500, 501, 515, 525, 527, 529, 535, 536, 546, 547, 551, 561, 563, 566, 573,
             577, 580, 585, 590, 597, 598, 599, 608, 614, 616, 632, 640, 644, 651, 659, 662, 669, 679, 681, 684, 685, 699, 706, 710, 724, 734, 740, 742, 743, 744, 745, 757, 763, 765, 768, 771, 775, 784, 786, 788, 789, 796, 798, 802, 806, 810, 813, 816, 819, 822, 836, 838, 840, 842, 852, 853, 859, 861, 865, 867, 869, 871, 872, 878, 879, 880, 881, 884, 887, 891, 895, 898, 903, 905, 923, 924, 925, 942, 943, 947, 958, 963, 964, 969, 977, 981, 985, 990, 991, 997, 998, 1000, 1001, 1002, 1015, 1017, 1021, 1025, 1029, 1035, 1036, 1037, 1041, 1043, 1050, 1056, 1060, 1061, 1064, 1066, 1070, 1071, 1072, 1075, 1079, 1081, 1084, 1088, 1090, 1097, 1099, 1102, 1106, 1107, 1112, 1115, 1117, 1120, 1125, 1128, 1130, 1132]

v100_easy_heavy = [1536, 11, 1549, 13, 17, 1559, 24, 29, 1566, 1567, 31, 33, 39, 1576, 47, 49, 1589, 1592, 1594, 1601, 1607, 1608, 1614, 1619, 1625, 1631, 1632, 1637, 1641, 1652, 1655, 1149, 1156, 1162, 1164, 1193, 1197, 1198, 1709, 1201, 1714, 1206, 1211, 1724, 1730, 1732, 1733, 1231, 1748, 1253,
                   1768, 1259, 1261, 1273, 1800, 1289, 276, 282, 283, 1309, 1825, 1331, 1843, 1348, 1860, 1865, 1362, 1374, 1376, 1389, 1915, 1415, 1417, 1420, 1940, 1432, 1439, 1958, 1448, 1451, 1454, 1455, 1968, 1458, 1972, 1463, 1465, 1470, 1986, 1483, 1487, 1493, 1498, 1507, 1514, 1515, 2031, 1521, 1528, 2042, 1535]

no_need_all = [1248, 1665, 2082, 1603, 1957, 1830, 1320, 1290, 1804, 1389, 1838, 1454, 2037, 1271, 1944]
no_need_cv = [1665, 1282, 1411, 1541, 1414, 1544, 1290, 1804, 1805, 1550, 1295, 1421, 1935, 1301, 1429, 1944, 1913, 1306, 1819, 1564, 1312, 1957, 1574, 1830, 1320, 1577, 1832, 2093, 1838, 1454, 1455, 1329, 1971, 2103, 2105, 1978, 1986, 579, 195, 1349, 1477, 1478, 1227, 1612, 1613, 1998, 1999, 1489, 1372, 1505, 1762, 1890, 2018, 998, 1383, 2028, 2037, 1526, 1271, 2041, 1914, 1663]
no_need_sv = [1922, 1420, 1425, 1940, 1816, 1819, 1185, 2084, 1576, 1320, 1448, 1453, 1969, 1843, 1846, 1974, 1594, 1986, 1478, 1481, 2005, 2018, 1383, 1384, 1389, 1262, 2034, 1268, 1913, 1531]
no_need_bank = [2048, 1538, 1539, 2055, 1418, 1291, 1424, 1682, 1299, 1300, 1561, 1532, 1444, 1573, 1575, 1450, 1452, 1456, 1585, 1457, 1203, 1458, 2101, 1591, 1464, 2106, 1472, 1473, 1985, 1475, 1226, 1355, 1356, 1229, 1357, 1360, 2000, 1492, 1622, 1369, 1498, 1371, 1499, 1246, 1247, 2015, 1633, 1378, 1379, 1380, 1382, 1510, 1512, 1516, 1774, 1518, 1519, 1393, 1395, 2035, 1781, 2042, 1404, 1279]

# sota_easy_ours_hard = {1665, 1541, 2054, 1287, 1805, 1298, 1913, 1306, 1454, 1969, 1604, 1478, 1998, 1878, 1505, 2018, 2023, 1389, 2037, 2041}
sota_easy_ours_hard = [1665, 1922, 1411, 1541, 2054, 1287, 1802, 1931, 1805, 1550, 1552, 1298, 1555, 1810, 1811, 1944, 1306, 2041, 1572, 1957, 1449, 2092, 1454, 1327, 1455, 1969, 1845, 2103, 1981, 1986, 1603, 1604, 451, 1606, 1478, 1994, 1612, 1998, 2001, 1878, 1879, 1372, 1505, 1890, 2018, 1893, 998, 2023, 1389, 1520, 2037, 1269, 1271, 1913, 1276, 1533, 1663]
#contain cross time
sota_easy_ours_hard = {1664, 1665, 1538, 1541, 2054, 2055, 1287, 1291, 1420, 1423, 1424, 1425, 1940, 1561, 1306, 1433, 2041, 2042, 1444, 1576, 1450, 1452, 1839, 1585, 1457, 1203, 1843, 1458, 1969, 1591, 1974, 1465, 1597, 1471, 1481, 1356, 1357, 1998, 1615, 1231, 1360, 1621, 2005, 1369, 1499, 1372, 2015, 1760, 1888, 1890, 1379, 1380, 1505, 2018, 1639, 2023, 1257, 1518, 1519, 1520, 2035, 1781, 1525, 1913, 1658, 1659, 1404, 1663}
# no cross time
sota_easy_ours_hard = {1538, 1541, 2054, 2055, 1287, 1291, 1420, 1423, 1424, 1425, 1561, 1306, 1433, 1444, 1576, 1450, 1452, 1585, 1457, 1203, 1458, 1969, 1974, 1591, 1465, 1597, 1471, 1481, 1356, 1357, 1998, 1615, 1231, 1360, 1621, 2005, 1369, 1499, 1372, 2015, 1760, 1505, 2018, 1379, 1380, 1639, 2023, 1257, 1518, 1519, 1520, 2035, 1781, 1525, 2041, 2042, 1404}
# sota easy and sota cveasy_svhard
sota_easy_ours_hard = {1536, 1538, 1541, 2054, 2055, 11, 1549, 1550, 13, 1552, 17, 1555, 1559, 24, 1561, 29, 1566, 1567, 31, 33, 1572, 1573, 1575, 1576, 39, 1578, 1580, 47, 1585, 49, 1588, 2101, 1589, 2103, 1591, 1593, 1592, 1612, 1614, 1619, 1622, 1625, 1631, 1633, 1637, 1641, 1149, 1156, 1162, 1163, 1164, 1682, 1702, 1197, 1198, 1709, 1201, 1714, 1203, 1206, 1211, 1724, 1730, 1732, 1226, 1229, 1748, 1246, 1247, 1250, 1764, 1253, 1766, 1768, 1257, 1261, 1262, 1774, 1269, 1781, 1271, 1273, 1274, 1276, 1279, 1287, 1800, 1289, 1291, 1298, 1299, 1300, 276, 1302, 1306, 282, 283, 1320, 1326, 1327, 1341, 1348, 1860, 1865, 1356, 1357, 1360, 1362, 1369, 1372, 1374, 1376, 1378, 1379, 1380, 1389, 1395, 1909, 1404, 1411, 1415, 1417, 1418, 1420, 1422, 1423, 1424, 1425, 1428, 1430, 1432, 1433, 1438, 1439, 1444, 1957, 1958, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1968, 1969, 1457, 1458, 1972, 1974, 1463, 1464, 1979, 1981, 1470, 1471, 1472, 1985, 1986, 451, 1478, 1480, 1481, 1994, 1483, 1998, 2001, 2005, 1493, 1495, 1498, 1499, 2015, 1505, 2018, 1507, 998, 2023, 1510, 1514, 1515, 1518, 1519, 1520, 2031, 2034, 2035, 1521, 2037, 1525, 1528, 2041, 2042, 1533, 1535}
# no need sota or heavy easy or ours heavy_hard
sota_easy_ours_hard = {1536, 1538, 1541, 2054, 2055, 2057, 2058, 11, 1549, 1550, 13, 1552, 2062, 1551, 1555, 17, 2064, 1559, 24, 1561, 1563, 1564, 29, 1566, 1567, 31, 33, 1572, 1573, 2084, 1575, 1576, 2086, 1578, 39, 1577, 1580, 2094, 47, 1585, 49, 2045, 1588, 2101, 1589, 2103, 1591, 1593, 1592, 2107, 1612, 1614, 1619, 1622, 1625, 1631, 1633, 1637, 1641, 1149, 1156, 1162, 1163, 1164, 1682, 1702, 1197, 1198, 1709, 1201, 1714, 1203, 1206, 1211, 1724, 1730, 1732, 1226, 1229, 1232, 1748, 1246, 1247, 1250, 1764, 1253, 1766, 1768, 1257, 1261, 1262, 1774, 1269, 1781, 1271, 1270, 1273, 1274, 1276, 1279, 1281, 1287, 1800, 1289, 1291, 1295, 1298, 1299, 1300, 276, 1302, 1306, 282, 283, 1320, 1323, 1326, 1327, 1841, 1341, 1348, 1860, 1865, 1356, 1357, 1360, 1362, 1369, 1372, 1374, 1376, 1378, 1379, 1380, 1389, 1390, 1906, 1395, 1396, 1909, 1913, 1914, 1401, 1404, 1402, 1411, 1415, 1416, 1417, 1418, 1420, 1422, 1423, 1424, 1425, 1935, 1428, 1430, 1432, 1433, 1945, 1434, 1438, 1439, 1444, 1957, 1958, 1449, 1450, 1961, 1452, 1453, 1454, 1455, 1451, 1969, 1457, 1458, 1968, 1971, 1974, 1972, 1464, 1463, 1979, 1981, 1470, 1471, 1472, 1985, 1986, 451, 1478, 1990, 1480, 1481, 1994, 1483, 1998, 2001, 2005, 1493, 1495, 1498, 1499, 2010, 2015, 1505, 2018, 1507, 2021, 998, 2023, 1510, 1514, 1515, 1518, 1519, 1520, 2031, 2034, 2035, 1521, 2037, 1525, 1528, 2041, 2042, 1533, 1535}

class CP2000(BaseImageDataset):
    """
    CP2000
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities:   (+1 for background)
    # images:   (train) +   (query) +   (gallery)
    # id structure:
    """
    dataset_dir = 'CP2000'
    dataset_name = 'CP2000'
    logger = logging.getLogger("CLIP-ReID.dataset")

    def __init__(self, root=r'../datasets/', verbose=True, pid_begin=0, **kwargs):
        self.cfg = kwargs.pop('cfg', None)
        super().__init__(**kwargs)

        if self.cfg is None:
            self.use_text = False
            self.use_attr = False
            self.text_prompt = ""
            self.text_type = 'captions'
            self.text_format = 'hybird'
            split_version = 100
        else:
            self.use_text = self.cfg.MODEL.USE_TEXT
            self.text_prompt = 'X ' * self.cfg.MODEL.TEXT_PROMPT if self.cfg.MODEL.TEXT_PROMPT > 0 else ""
            self.use_attr = self.cfg.MODEL.USE_ATTR
            self.text_type = self.cfg.MODEL.TEXT_TYPE
            self.text_format = self.cfg.MODEL.TEXT_FORMAT
            split_version = self.cfg.DATASETS.VERSION

        self.split_version = split_version
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, f'split_v{split_version}', f'train_v{split_version}.txt')
        self.query_dir = osp.join(self.dataset_dir, f'split_v{split_version}', f'query_v{split_version}.txt')
        self.gallery_dir = osp.join(self.dataset_dir, f'split_v{split_version}', f'gallery_v{split_version}.txt')
        self.logger.info(colored(f"Dataset split version: {split_version}", 'green'))

        self.use_attrs = ['Glasses', 'Holding Phone', 'Head Accessories', 'Accessories', 'Pose', 'Upper Clothing',
                          'Upper Color', 'Upper Style', 'Lower Clothing', 'Lower Color', 'Lower Style', 'Feet']

        self._check_before_run()
        self.pid_begin = pid_begin

        # attribute setting loading
        if self.use_attr:
            attribute_path = osp.join(self.dataset_dir, 'attributes_labeled_qwen.json')
            attribute_template_path = osp.join(self.dataset_dir, 'attr_translation_idx.json')
            # self.delete_attr = ['View','Illumination']
            self.attribute_anno = json.load(open(attribute_path, 'r'))
            self.attribute_map = json.load(open(attribute_template_path, 'r'))
            self.attribute_num_classes = {k: len(v) for k, v in self.attribute_map.items() if k in self.use_attrs}
            self.attribute_names = sorted(self.attribute_num_classes.keys())
            self.attributes = self.parse_attributes(self.attribute_anno)

        else:
            # self.delete_attr = {}
            self.attribute_anno = {}
            self.attribute_map = {}
            self.attribute_names = []
            self.attribute_num_classes = {}
            self.attributes = {}
        # text setting loading
        if self.use_text:
            if self.text_type == 'captions':
                text_path = osp.join(self.dataset_dir, 'attribute_captions_doubao.json')
                texts = json.load(open(text_path, 'r'))
                self.texts = self.parse_texts(texts)
            else:
                attribute_path = osp.join(self.dataset_dir, 'attributes_labeled_qwen.json')
                attribute_template_path = osp.join(self.dataset_dir, 'attr_translation_idx.json')
                attribute_anno = json.load(open(attribute_path, 'r'))
                attribute_map = json.load(open(attribute_template_path, 'r'))
                attribute_num_classes = {k: len(v) for k, v in attribute_map.items() if k in self.use_attrs}
                attribute_names = sorted(attribute_num_classes.keys())
                self.texts = self.parse_texts_clean(attribute_anno, attribute_names)
                # self.logger.info(colored(f"Attributes ({len(attribute_names)}): {attribute_names}", 'green'))
        else:
            self.texts = None

        self.train, self.query, self.gallery = self.make_dataset(train_dir=self.train_dir,
                                                                 query_dir=self.query_dir,
                                                                 gallery_dir=self.gallery_dir)

        if verbose:
            self.logger.info(f"=> {self.dataset_name} loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train, print_cam=True)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query, print_cam=True)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery, print_cam=True)

    def parse_texts_clean(self, attribute_anno=None, attribute_names=None):

        text_prefix = 'An image of a person with the following attributes: '

        pid_factory = {}
        for key, attrs in attribute_anno.items():
            parts = key.split('_')
            pid = int(parts[0])
            viewid = int(parts[1])
            timeid = int(parts[2])
            if pid not in pid_factory:
                pid_factory[pid] = {}
            if viewid not in pid_factory[pid]:
                pid_factory[pid][viewid] = {}
            attr_for_test = ''
            for attr in attribute_names:
                attr_for_test += f"{attr.replace(' ', '_')} is {attrs[attr]}, "
            pid_factory[pid][viewid][timeid] = f'{attr_for_test.strip()[:-1]}.'
            # pid_factory[pid][viewid][timeid] = f'{text_prefix} {attr_for_test.strip()[:-1]}.'

        pid_factory = self.repair_factory(pid_factory)
        return pid_factory

    def parse_texts(self, texts):
        pid_factory = {}
        for key, text in texts.items():
            parts = key.split('_')
            pid = int(parts[0])
            viewid = int(parts[1])
            timeid = int(parts[2])
            if pid not in pid_factory:
                pid_factory[pid] = {}
            if viewid not in pid_factory[pid]:
                pid_factory[pid][viewid] = {}
            pid_factory[pid][viewid][timeid] = text

        pid_factory = self.repair_factory(pid_factory)
        return pid_factory

    def parse_attributes(self, attributes=None, delete_keys=[]):
        pid_factory = {}
        for key, attrs in attributes.items():
            attr_label = []
            for attr in self.attribute_names:
                if attr in delete_keys:
                    continue

                value = attrs.get(attr, 'unknown')
                # if value =="Sweatshirts":
                #     value = "jacket"
                attr_label.append(self.attribute_map[attr][value])
            parts = key.split('_')
            pid = int(parts[0])
            viewid = int(parts[1])
            timeid = int(parts[2])
            if pid not in pid_factory:
                pid_factory[pid] = {}
            if viewid not in pid_factory[pid]:
                pid_factory[pid][viewid] = {}
            # pid_attributes[key] = torch.tensor(attr_label, dtype=torch.int64)
            pid_factory[pid][viewid][timeid] = torch.tensor(attr_label, dtype=torch.int64)
        pid_factory = self.repair_factory(pid_factory)

        return pid_factory

    def repair_factory(self, pid_factory):
        # 容错
        for pid in pid_factory:
            if 0 not in pid_factory[pid]:
                pid_factory[pid][0] = copy.deepcopy(pid_factory[pid][1])
            elif 1 not in pid_factory[pid]:
                pid_factory[pid][1] = copy.deepcopy(pid_factory[pid][0])

            for viewid in range(2):
                if 0 not in pid_factory[pid][viewid]:
                    pid_factory[pid][viewid][0] = copy.deepcopy(pid_factory[pid][viewid][1])
                elif 1 not in pid_factory[pid][viewid]:
                    pid_factory[pid][viewid][1] = copy.deepcopy(pid_factory[pid][viewid][0])
        return pid_factory

    def get_item_per_img(self, factory, pid, viewid, timeid):
        # key = f"{pid}_{viewid}_{timeid}"
        # text = texts.get(key, None)
        # if text is not None:
        #     key_brother = f"{pid}_{viewid}_{0 if timeid == 1 else 1}"
        #     text = texts.get(key_brother, None)
        item = factory[pid][viewid][timeid]
        return item

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        query = self._process_dir(query_dir, train=False, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')
        return train, query, gallery

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(
                "'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(
                "'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, train=False, mode='mix', gallery_pids=None):
        # img_paths = glob.glob(osp.join(dir_path, '**', '*.jpg'), recursive=True)
        root = Path(dir_path)
        split_version = f"images_v{self.split_version}" if self.split_version >= 100 else 'images'
        data_dir = root.parent.parent / split_version
        with open(root, 'r') as f:
            data = f.read().strip().split('\n')
        img_paths = [str(data_dir / line) for line in data]

        pid_container = set()
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            try:
                pid = int(img_name.split('_')[0])
            except Exception as e:
                self.logger.error(f"Error parsing PID from image name '{img_name}': {e}")
                raise
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2index = {pid: index for index, pid in enumerate(pid_container)}
        dataset = []
        views = {'aerial': 0, 'ground': 0}
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            parts = img_name.split('_')
            pid = int(parts[0])
            camid = int(parts[1])
            timeid = int(parts[2])
            sceneid = int(parts[3])
            # if pid in sota_easy_ours_hard:
            #     continue
            if camid < 23:
                views['ground'] += 1
                viewid = 0  # ground view
                if mode == 'aerial':
                    continue
                if mode == 'aerial-ground':
                    camid = 1
            else:   # aerial view
                views['aerial'] += 1
                viewid = 1
                if mode == 'ground':
                    continue
                if mode == 'aerial-ground':
                    camid = 2

            camid -= 1  # index starts from 0
            pid_idx = pid2index[pid] if train else pid

            if gallery_pids is not None and pid_idx not in gallery_pids:
                continue

            # if pid in v501_easy[:150]:
            #     continue
            # platform_name = {0: "UAV", 1: "CCTV"}[viewid]
            # platform_name = "X"
            use_text = self.use_text
            if train:
                text_anno = f"A photo of a {self.text_prompt} person with the following attributes: {self.get_item_per_img(self.texts, pid, 0, 1)}" if use_text else None
            else:
                text_anno = f"A photo of a {self.text_prompt} person." if use_text else None

            use_attr = self.use_attr
            # attr_anno = self.get_item_per_img(self.attributes, pid, 0, 1) if use_attr else None
            attr_anno = None if use_attr else None

            # if self.use_text:
            #     platform_name = {0: "UAV", 1: "CCTV"}[viewid]
            #     if train:
            #         text_anno = f"An image of a {self.text_prompt} in the {platform_name} platform, capturing natural colors and fine details: {self.get_item_per_img(self.texts, pid, 0, 1)}"
            #         attr_anno = self.get_item_per_img(self.attributes, pid, viewid, 1) if self.use_attr else None
            #     else:
            #         text_anno = f"An image of a {self.text_prompt} in the {platform_name} platform. capturing natural colors and fine details are unkown."
            #         attr_anno = None
            # else:
            #     text_anno = None
            #     if train:
            #         attr_anno = self.get_item_per_img(self.attributes, pid, 0, 1) if self.use_attr else None
            #     else:
            #         attr_anno = None
            dataset.append((img_path, self.pid_begin + pid_idx, camid, viewid, timeid, [text_anno, attr_anno]))

        self.logger.info(f'{dir_path}: {views}')
        return dataset


class CP2000_ALL(CP2000):
    dataset_name = 'CP2000_all'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')
        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='mix', gallery_pids=gallery_pids)
        # query_new = [entry for entry in query]
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')
        query = [entry for entry in query if entry[1] not in sota_easy_ours_hard]
        return train, query, gallery


class CP2000_GA(CP2000):
    dataset_name = 'CP2000_ga'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='ground', gallery_pids=gallery_pids)
        query = [entry for entry in query if entry[1] not in sota_easy_ours_hard]
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')
        
        return train, query, gallery


class CP2000_AG(CP2000):
    dataset_name = 'CP2000_ag'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        # query_dir = osp.join(self.dataset_dir, 'query')
        # gallery_dir = osp.join(self.dataset_dir, 'gallery')
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='ground')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)
        query = [entry for entry in query if entry[1] not in sota_easy_ours_hard]
        return train, query, gallery


class CP2000_AA(CP2000):
    dataset_dir = 'CP2000'
    dataset_name = 'CP2000_aa'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        # query_dir = osp.join(self.dataset_dir, 'quer')
        # gallery_dir = osp.join(self.dataset_dir, 'gallery')
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='aerial', gallery_pids=gallery_pids)
        # query_new = query
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')
        query = [entry for entry in query if entry[1] not in sota_easy_ours_hard]
        return train, query, gallery


class CP2000_GG(CP2000):
    dataset_dir = 'CP2000'
    dataset_name = 'CP2000_gg'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        # query_dir = osp.join(self.dataset_dir, 'query')
        # gallery_dir = osp.join(self.dataset_dir, 'gallery')
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='ground')
        gallery_pids = {item[1] for item in gallery}

        query = self._process_dir(query_dir, train=False, mode='ground', gallery_pids=gallery_pids)
        # query_new = [entry for entry in query]
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')
        query = [entry for entry in query if entry[1] not in sota_easy_ours_hard]
        return train, query, gallery


class CP2000_AGAG(CP2000):
    dataset_dir = 'CP2000'
    dataset_name = 'CP2000_agag'

    def make_dataset(self, train_dir, query_dir, gallery_dir):

        train = self._process_dir(train_dir, train=True, mode='mix')
        query = self._process_dir(query_dir, train=False, mode='aerial-ground')
        gallery = self._process_dir(gallery_dir, train=False, mode='aerial-ground')

        # query_new = [entry for entry in query if entry[1] not in cross_view_me_easy]
        # query_new = query
        # self.logger.info(f'query: {len(query)} -> {len(query_new)}')
        query = [entry for entry in query if entry[1] not in sota_easy_ours_hard]
        return train, query, gallery


class CP2000_AAGG(CP2000):
    dataset_name = 'CP2000_aagg'

    def make_dataset(self, train_dir, query_dir, gallery_dir):
        train = self._process_dir(train_dir, train=True, mode='mix')
        gallery = self._process_dir(gallery_dir, train=False, mode='mix')
        gallery_pids = {item[1] for item in gallery}
        query = self._process_dir(query_dir, train=False, mode='mix', gallery_pids=gallery_pids)
        query = [entry for entry in query if entry[1] not in sota_easy_ours_hard]
        return train, query, gallery


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    print(f"{os.getcwd()}")
    from utils.logger import setup_logger
    logger = setup_logger("CLIP-ReID", save_dir=None, if_train=True)
    dataset = CP2000_ALL()
    dataset_ag = CP2000_AG()
    dataset_ga = CP2000_GA()
    dataset_aa = CP2000_AA()
    dataset_gg = CP2000_GG()
    dataset_agag = CP2000_AGAG()
