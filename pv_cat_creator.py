#############################################################
#Partiview Catalogue Creator v1.1
#Developed by D Eden
#
#Designed to convert catalogues of format (l,b,Dist)
#to Partiview coordinate system.
#
#
#
#Version History
#v1.1 Produces .speck file of Mopra Cloud Catalogue
#v1.0 Reads in HII region catalogue of Anderson et al. (2014)

#In Development
#Write correctly to .speck file with required preample
#Read in any catalogue
#############################################################
#Imports pandas for csv files
import pandas as pd

#Allows for massive catalgoues, if using bigger catalogue (such as Gaia), may need to increase.
pd.options.display.max_rows = 99999

#Initialise astropy
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord, Distance, CartesianRepresentation
from astropy import units as u

#Intiliase numpy
import numpy

#Read in catalogues
mopra_clouds = pd.read_csv('/Users/kyriakos.trakakis/Downloads/CHIMPS_COHRS/13CO/FGN_13CO_017_047_vrms_3_mass_catalogue_short.csv')
mopraCoords=SkyCoord(mopra_clouds['gal_long (deg)'], mopra_clouds['gal_lat (deg)'], unit=(u.degree,u.degree), distance=Distance(1000*(mopra_clouds['distance (pc)']),u.pc), frame='galactic')

#Add the Cartesian, Galactic coordinates to the speck file
df = pd.DataFrame(mopra_clouds)

x_gal_coord = mopraCoords.galactic.represent_as(CartesianRepresentation).x.to(u.pc).value
df.insert(0,"x_coord", x_gal_coord)

y_gal_coord = mopraCoords.galactic.represent_as(CartesianRepresentation).y.to(u.pc).value
df.insert(1,"y_coord", y_gal_coord)

z_gal_coord = mopraCoords.galactic.represent_as(CartesianRepresentation).z.to(u.pc).value
df.insert(2,"z_coord", z_gal_coord)

#Remove source name
#del df["_idx"]

#Produces speck file
df.to_csv('FGN_13CO_017_047_clusters_short.speck', header='headerText', index=None, sep=' ', mode='a')

print(df)
