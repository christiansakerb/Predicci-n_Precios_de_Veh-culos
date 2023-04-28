#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predic_price(marca,modelo,millas,estado_uso,ano):
    return 10


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('por favor insertar la marca,modelo,millas,estado uso y año')
    else:
        url = sys.argv[1]
        p1 = predic_price(url)
        print(url)
        print('La probabilidad del precio del carro es: ', p1)
        