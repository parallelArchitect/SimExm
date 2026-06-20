# Provided under BSD license
# Copyright (c) 2017, Jeremy Wohlwend
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of SimExm nor the names of its contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL JEREMY WOHLWEND BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
fluors.py

The Fluorophore class and a subclass for each type (for easy data access and storage)

Ported to Python 3: this file had no Python 2-only syntax (no print
statements, no xrange, no np.int). Two real bugs were fixed during
the port, unrelated to Python 2/3 compatibility:
  - get_emission_file() and get_excitation_file() opened files with
    open() and never closed them. Now use 'with' blocks, which close
    the file automatically even if parsing raises an exception.
  - The original split data on "\r" only (classic Mac-style line
    endings). If these data files are ever re-saved on a different
    OS, that produces one giant unsplit string instead of multiple
    rows -- a silent data-corruption risk independent of Python
    version. Now uses str.splitlines(), which correctly handles
    \r, \n, and \r\n.
"""

import os
import numpy as np

class Fluorset: 
    """
    Interface to access fluorophore data. The data is located in the fluordata folder.
    This is a pre-initialized dataset which is used to query flurophore parameters in the labeling simualtion.
    """
    def __init__(self):
        """
        Init method, hard-coded.
        all_fluors (list of FLuorophore objects) : list of all fluorophore instances in fluors.py
        """
        self.all_fluors = [Alexa350(), Alexa790(), ATTO390(), ATTO425(), ATTO430LS(),\
                    ATTO465(), ATTO488(), ATTO490LS(), ATTO550(), ATTO647N(), ATTO700()]

    def get_all_fluorophores_types(self):
        """Returns a list of flurophore types as a list of strings"""
        return [fluor.get_name() for fluor in self.all_fluors]


    def get_fluor(self, name):
        """
        Returns the desired Fluorophore object. See fluors.py
        
        Args:
            name: string
                the name of the fluorophore to query
        Returns:
            fluor: Fluorophore object
                the desired fluorophore object
        """
        fluors = self.get_all_fluorophores_types()
        index = fluors.index(name)
        fluor = self.all_fluors[index]
        return fluor


class Fluorophore:
    """
    Interface to interace with a fluorophore's data.
    Instances of the FLuorohore class are used to enumerate fluorophore parameters,
    some of which are hard-coded in this file.

    name: string 
        Name given to the fluorophore
    xls: string
        Path to xls file containing fluorophore data
    excitation: string
        Path to file containing excitation data
    emission: string
        Path to file containing excitation data
    extinction_coefficient: int
        Coefficient of exctinction
    quantum_yield: float
        Value of the quantum yeild
    source: string
        Source where the data can be obtained / verified
    comments: string 
        Additonal comments for the user 
    """
    name = ""
    xls = ""
    quantum_yield = 0.0
    extinction_coefficient = 0.0
    emission = ""
    excitation = ""
    source = ""
    comments = ""


    def get_name(self):
        """Returns the (string) type of the fluoropore"""
        return self.name

    def get_quantum_yield(self):
        """Returns the (float) quantum yield of the fluorophore"""
        return float(self.quantum_yield)

    def get_extinction_coefficient(self):
        """Returns the (float) extinction coefficient of the fluorophore"""
        return float(self.extinction_coefficient)

    def get_source(self):
        """Returns the (string) source, where information about the flurophore can be found"""
        return self.source

    def get_comments(self):
        """Returns (string) comments, addional information about the fluorophore"""
        return self.comments

    def get_emission_file(self):
        """Returns emission data as a float numpy array"""
        file_path = self.emission
        with open(file_path, 'r') as f:
            raw_data = f.read()
        lines = raw_data.splitlines()
        raw_data = np.array([s.split("\t") for s in lines], dtype=float)
        return raw_data

    def get_excitation_file(self):
        """Returns excitation data as a float numpy array"""
        file_path = self.excitation
        with open(file_path, 'r') as f:
            raw_data = f.read()
        lines = raw_data.splitlines()
        raw_data = np.array([s.split("\t") for s in lines], dtype=float)
        return raw_data

    def find_emission_peak(self):
        """Returns the emission peak for that fluorophore"""
        raw_data = self.get_emission_file()
        index = np.argmax(raw_data[:, 1])
        return raw_data[index, 0]
        

    def find_excitation(self, wavelength):
        """Returns the excitation value of the given fluorophore, 
        at the given (int) wavelenght in nm"""
        raw_data = self.get_excitation_file()
        data     = raw_data[:,1] / np.max(raw_data[:,1])
        data_min = np.min(raw_data[:,0])
        data_max = np.max(raw_data[:,0])
        interval = round(np.mean(np.diff(raw_data[:,0])),2)

        if (wavelength < data_min) or (wavelength > data_max):
            excitation = 0
        else:
            index      = int(np.floor((wavelength - data_min) / interval))
            index -= 1
            weight     = wavelength - (data_min + index * interval)
            excitation = weight * data[index + 2] + (1 - weight) * data[index + 1]

        return excitation

    def find_emission(self, laser_filter):
        """Returns the emission value of the given fluorophore, at the given minimum
        and maximum (int) wavelenghts in nm"""
        wavelength_min, wavelength_max = laser_filter
        raw_data = self.get_emission_file()
        data     = raw_data[:,1] / np.sum(raw_data[:,1])
        interval = round(np.mean(np.diff(raw_data[:,0])),2)
        data_min = np.min(raw_data[:,0])
        data_max = np.max(raw_data[:,0])
        emission = 0

        if wavelength_max < data_max:
            index = int(np.floor((wavelength_max - data_min) / interval))
        else:
            index = len(data) - 1

        index -= 1

        while (index >= 0) and (wavelength_min <= data_min + index * interval):
            emission = emission + data[index + 1]
            index = index - 1

        return emission


#Path to data
fluo_path = os.path.abspath(os.path.dirname(__file__)) + '/../fluordata/'

# List of fluorophores in database, as FLuorophore objects
# Alexa DYES


class Alexa350(Fluorophore):

    name = "Alexa350"
    xls = fluo_path  + "Alexa350.xls"
    excitation = fluo_path +"Alexa350_excitation.txt"
    emission = fluo_path +"Alexa350_emission.txt"
    extinction_coefficient = 19000
    quantum_yield = 0.25 
    source = "https://www.thermofisher.com/us/en/home/references/molecular-probes-the-handbook/fluorophores-and-their-amine-reactive-derivatives/alexa-fluor-dyes-spanning-the-visible-and-infrared-spectrum.html"
    comments = "The quantum yeild value is a guess, the actual figure is not provided."


class Alexa790(Fluorophore):

    name = "Alexa790"
    xls = fluo_path  + "Alexa790.xls"
    excitation = fluo_path +"Alexa790_excitation.txt"
    emission = fluo_path +"Alexa790_emission.txt"
    extinction_coefficient = 260000
    quantum_yield = 0.25 
    source = "https://www.thermofisher.com/us/en/home/references/molecular-probes-the-handbook/fluorophores-and-their-amine-reactive-derivatives/alexa-fluor-dyes-spanning-the-visible-and-infrared-spectrum.html"
    comments = "The quantum yeild value is a guess, the actual figure is not provided."


# ATTO DYES


class ATTO390(Fluorophore):

    name = "ATTO390"
    xls = fluo_path  + "ATTO390.xls"
    excitation = fluo_path +"ATTO390_excitation.txt"
    emission = fluo_path +"ATTO390_emission.txt"
    extinction_coefficient = 24000
    quantum_yield = 0.9 
    source = "http://www.atto-tec.com/fileadmin/user_upload/Katalog_Flyer_Support/ATTO_390.pdf"
    comments = ""


class ATTO425(Fluorophore):

    name = "ATTO425"
    xls = fluo_path  + "ATTO425.xls"
    excitation = fluo_path +"ATTO425_excitation.txt"
    emission = fluo_path +"ATTO425_emission.txt"
    extinction_coefficient = 45000
    quantum_yield = 0.9 
    source = "http://www.atto-tec.com/fileadmin/user_upload/Produktdatenblaetter/ATTO_425.pdf"
    comments = ""


class ATTO430LS(Fluorophore):

    name = "ATTO430LS"
    xls = fluo_path  + "ATTO430LS.xls"
    excitation = fluo_path +"ATTO430LS_excitation.txt"
    emission = fluo_path +"ATTO430LS_emission.txt"
    extinction_coefficient = 32000
    quantum_yield = 0.65 
    source = "http://www.atto-tec.com/fileadmin/user_upload/Katalog_Flyer_Support/ATTO_430LS.pdf"
    comments = ""


class ATTO465(Fluorophore):

    name = "ATTO465"
    xls = fluo_path  + "ATTO465.xls"
    excitation = fluo_path +"ATTO465_excitation.txt"
    emission = fluo_path +"ATTO465_emission.txt"
    extinction_coefficient = 75000
    quantum_yield = 0.75 
    source = "http://www.atto-tec.com/fileadmin/user_upload/Katalog_Flyer_Support/ATTO_465.pdf"
    comments = ""


class ATTO488(Fluorophore):

    name = "ATTO488"
    xls = fluo_path  + "ATTO488.xls"
    excitation = fluo_path +"ATTO488_excitation.txt"
    emission = fluo_path +"ATTO488_emission.txt"
    extinction_coefficient = 90000
    quantum_yield = 0.8 
    source = "http://www.atto-tec.com/fileadmin/user_upload/Katalog_Flyer_Support/ATTO_488.pdf"
    comments = ""


class ATTO490LS(Fluorophore):

    name = "ATTO490LS"
    xls = fluo_path  + "ATTO490LS.xls"
    excitation = fluo_path +"ATTO490LS_excitation.txt"
    emission = fluo_path +"ATTO490LS_emission.txt"
    extinction_coefficient = 40000
    quantum_yield = 0.3 
    source = "http://www.atto-tec.com/fileadmin/user_upload/Katalog_Flyer_Support/ATTO_490LS.pdf"
    comments = ""


class ATTO550(Fluorophore):

    name = "ATTO550"
    xls = fluo_path  + "ATTO550.xls"
    excitation = fluo_path +"ATTO550_excitation.txt"
    emission = fluo_path +"ATTO550_emission.txt"
    extinction_coefficient = 120000
    quantum_yield = 0.3 
    source = "http://www.atto-tec.com/fileadmin/user_upload/Katalog_Flyer_Support/ATTO_550.pdf"
    comments = ""


class ATTO647N(Fluorophore):

    name = "ATTO647N"
    xls = fluo_path  + "ATTO647N.xls"
    excitation = fluo_path +"ATTO647N_excitation.txt"
    emission = fluo_path +"ATTO647N_emission.txt"
    extinction_coefficient = 150000
    quantum_yield = 0.65 
    source = "http://www.atto-tec.com/fileadmin/user_upload/Katalog_Flyer_Support/ATTO_647N.pdf"
    comments = ""


class ATTO700(Fluorophore):

    name = "ATTO700"
    xls = fluo_path  + "ATTO700.xls"
    excitation = fluo_path +"ATTO700_excitation.txt"
    emission = fluo_path +"ATTO700_emission.txt"
    extinction_coefficient = 120000
    quantum_yield = 0.25 
    source = "http://www.atto-tec.com/fileadmin/user_upload/Katalog_Flyer_Support/ATTO_700.pdf"
    comments = ""
