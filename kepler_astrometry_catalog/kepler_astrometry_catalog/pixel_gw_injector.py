"""
gw_pixel_injector.py

Injects gravitational wave-induced astrometric deflections into Kepler pixel-level centroid data.

This script performs the following steps:
1. Loads raw centroids from an h5 file and converts pixel coordinates to sky coordinates using RaDec2Pix.
2. Projects sky coordinates to telescope frame using Kepler's pointing.
3. Calculates GW deflections using the estoiles.gw_source module in the telescope frame.
4. Applies deflections in the telescope frame and transforms back to sky coordinates.
5. Converts sky coordinates back into pixel coordinates (RaDec2Pix), to create fake centroids.

Intended for generating synthetic datasets with injected GW signals for testing astrometric analysis pipelines.

Created by Tal Adi
"""

import tables
import shutil
import os
import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from estoiles.gw_source import GWSource
from raDec2Pix import raDec2Pix
import estoiles.gw_calc
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from pathlib import Path

class PixelGWInjector:
    def __init__(self, h5file_path=None, gw_params=None, parallel=True):
        self.parallel = parallel
        # Read structure of the HDF5 file to determine available quarters, modules, and outputs
        if h5file_path is None:
            print("No HDF5 file path provided, exiting.")
            return
        self.structure = self.get_file_structure(h5file_path)
        # Store the HDF5 file path
        self.h5file_path = h5file_path
        # Temporary values for testing
        # quarter = 12
        # module = 13
        # output = 4
        # Kepler pointing coordinates
        # TODO: actual pointing coordinates change with time (seepointingModel in raDec2PixModels.py)
        DEFAULT_TELCOORD = SkyCoord(ra='19h22m40s', dec='+44:30:00', unit=(u.hourangle, u.deg), frame='icrs')
        # Default source coordinates for the gravitational wave source
        if gw_params is None:
            gw_params = {
                "freq": 5.0e-8 * u.Hz,
                "Mc": 5.0e6 * u.Msun,
                "q": 1,
                "dl": 0.2 * u.Mpc,
                "inc": 0.0 * u.deg,
                "psi": 0.0 * u.deg,
                "phi": None,
                "initial_phip": 0.0 * u.rad,
                "initial_phic": 0.0 * u.rad,
                "time": 0.*u.s,
                "telcoord": DEFAULT_TELCOORD.galactic,
                "sourcecoord": SkyCoord(l=0*u.deg,b=90*u.deg,frame='galactic'),
                "post_newtonian_orderlist": np.array([True, False, False, False, False])
            }
        self.gw_params = gw_params
        # Initialize the GWSource object with the default parameters
        gw_source = GWSource(**gw_params)
        
        # Iterate over the quarters, modules, and outputs in the structure
        for quarter, modules in self.structure.items():
            for module, outputs in modules.items():
                for output in outputs:
                    # Load the batch of stars from the HDF5 file and compute deflections
                    print(f"Processing quarter {quarter}, module {module}, output {output}")
                    # Load the batch of stars from the HDF5 file and compute deflections using the new method
                    kicids, time, deflected_x, deflected_y = self.generate_deflected_centroids(
                        h5file_path, quarter, module, output, gw_source)
                    # Save the deflected centroids to a new HDF5 file
                    self.save_to_h5file(kicids, deflected_x, deflected_y, time, quarter, module, output, gw_params)

                    # Uncomment the following lines to print the results for debugging
                    # self.gw_source = gw_source
                    # self.quarter = quarter
                    # self.module = module
                    # self.output = output
                    # self.kicids = kicids
                    # self.def_x = deflected_x
                    # self.def_y = deflected_y
                    # self.time = time

    def generate_deflected_centroids(self, h5file_path, quarter, module, output, gw_source):
        # Load the batch of stars from the HDF5 file
        kicids, x, y, time = self.load_batch(h5file_path, quarter, module, output)
        # Calculate the mean x and y values and the mean time value
        mean_x = np.mean(x, axis=1)
        mean_y = np.mean(y, axis=1)
        mean_time = np.mean(time)  # in seconds
        # Generate the star catalog from the loaded data, using the *mean* x and y pixel coordinates
        stars_skycoord, time_obj = self.stars_skycoord_gen(mean_x, mean_y, mean_time, module, output)
        # Calculate deflections for the *mean* star coordinates in the telescope frame for each star time array
        if self.parallel:
            deflected_mean_x, deflected_mean_y = self.calculate_deflections_parallel(gw_source, self.gw_params['telcoord'],
                                                                                   stars_skycoord, time, time_obj,
                                                                                   module, output)
            non_def_mean_x, non_def_mean_y = self.calculate_deflections(None, self.gw_params['telcoord'],
                                                                       stars_skycoord, time, time_obj,
                                                                       module, output)
        else:
            deflected_mean_x, deflected_mean_y = self.calculate_deflections(gw_source, self.gw_params['telcoord'],
                                                                          stars_skycoord, time, time_obj,
                                                                          module, output)
            non_def_mean_x, non_def_mean_y = self.calculate_deflections(None, self.gw_params['telcoord'],
                                                                       stars_skycoord, time, time_obj,
                                                                       module, output)
        # Remove the mean values from the deflected coordinates to get the deflections
        dx = deflected_mean_x - non_def_mean_x    # shape (N_stars, N_times)
        dy = deflected_mean_y - non_def_mean_y    # shape (N_stars, N_times)
        # Add deflections to the original x and y centroids time-series
        deflected_x = x + dx    # shape (N_stars, N_times)
        deflected_y = y + dy    # shape (N_stars, N_times)
        return kicids, time, deflected_x, deflected_y



    @staticmethod
    def get_file_structure(h5file_path):
        """
        Inspect the structure of the HDF5 file to determine which quarters (Q* groups),
        and within each quarter which modules and outputs are present.

        Args:
            h5file_path: Path to the HDF5 file.

        Returns:
            A dictionary in the format:
            {
                'Q12': {
                    2: [1, 3],
                    13: [4]
                },
                ...
            }
        """
        if not Path(h5file_path).exists():
            raise FileNotFoundError(f"HDF5 file {h5file_path} does not exist.")
        # Open the HDF5 file and read the groups to determine the structure
        structure = {}
        with tables.open_file(h5file_path, 'r') as f:
            for group in f.root._v_groups.values():
                if group._v_name.startswith('Q'):
                    quarter = group._v_name
                    pos = group.pos
                    modules = pos.read(field="module")
                    outputs = pos.read(field="output")
                    mod_out_dict = {}
                    for i in range(modules.shape[0]):
                        mod_unique = np.unique(modules[i])
                        out_unique = np.unique(outputs[i])
                        for mod in mod_unique:
                            if mod not in mod_out_dict:
                                mod_out_dict[mod] = set()
                            mod_out_dict[mod].update(out_unique)
                    # Convert sets to sorted lists
                    structure[quarter] = {mod: sorted(list(outs)) for mod, outs in mod_out_dict.items()}
        return structure


    @staticmethod
    def load_batch(h5file_path, quarter, module, output):
        """
        Load a batch of stars from the HDF5 file for the specified quarter, module, and output.
    
        Key arguments:
            - h5file_path: Path to the HDF5 file containing the star data.
            - quarter: Quarter number to filter the stars.
            - module: Module number to filter the stars.
            - output: Output number to filter the stars.
        Returns:
            - kicids: Array of KIC IDs for the selected stars.
            - x: Array of x pixel coordinates for the selected stars.
            - y: Array of y pixel coordinates for the selected stars.
            - time: Array of observation times in seconds for the selected stars.
        """
        print(f"loading from {h5file_path} quarter {quarter}, module {module}, output {output}")
        # Check if the file exists
        if not Path(h5file_path).exists():
            raise FileNotFoundError(f"HDF5 file {h5file_path} does not exist.")
        # Open the HDF5 file and read the relevant group for the specified quarter
        with tables.open_file(h5file_path, 'r') as f:
            group = getattr(f.root, f"{quarter}")
            pos = group.pos
            kicids = group.kicid.read()

            modules = pos.read(field="module")
            outputs = pos.read(field="output")
            times = pos.read(field="time")  # in seconds
            x = pos.read(field="x") # in pixels
            y = pos.read(field="y") # in pixels

            selected_indices = []

            for i in range(modules.shape[0]):
                mod = modules[i]
                out = outputs[i]
                valid = ~np.isnan(x[i]) & ~np.isnan(y[i])

                if np.any((mod[valid] == module) & (out[valid] == output)):
                    selected_indices.append(i)

            if not selected_indices:
                print(f"No stars found for quarter {quarter}, module {module}, output {output}")
                return

            selected_indices = np.array(selected_indices)
            kicids = kicids[selected_indices]
            time = times[selected_indices]
            x = x[selected_indices]
            y = y[selected_indices]
            return kicids, x, y, time


    @staticmethod
    def get_raDec2Pix(ra, dec, time_obj):
        """
        Convert RA/Dec coordinates to pixel coordinates using the raDec2Pix module.
        Parameters:
        - ra: Right Ascension in degrees (float or array-like)
        - dec: Declination in degrees (float or array-like)
        - time: Astropy Time object representing the observation time
        Returns:
        - m: Module number (int or array-like)
        - o: Output number (int or array-like)
        - r: Row pixel coordinates (int or array-like)
        - c: Column pixel coordinates (int or array-like)
        """
        rdp = raDec2Pix.raDec2PixClass()
        m, o, r, c = rdp.ra_dec_2_pix(ra, dec, time_obj.mjd)
        return m, o, r, c


    @staticmethod
    def get_Pix2RaDec(module, output, row, column, time_obj, aberrate=True):
        """
        Convert pixel coordinates to RA/Dec using the raDec2Pix module.
        
        Key arguments:
            - module: Module number (int or array-like)
            - output: Output number (int or array-like)
            - row: Row pixel coordinates (int or array-like)
            - column: Column pixel coordinates (int or array-like)
            - time: Astropy Time object representing the observation time
            - aberrate: Boolean flag to indicate whether to apply aberration correction (default is True)
        Returns:
            - ra: Right Ascension in degrees (float or array-like)
            - dec: Declination in degrees (float or array-like)
        """
        rdp = raDec2Pix.raDec2PixClass()
        ra, dec = rdp.pix_2_ra_dec(
            module, output, row, column, time_obj.mjd, aberrateFlag=aberrate
        )
        return ra, dec


    @staticmethod
    def stars_skycoord_gen(x, y, time, module, output):
        """
        Generate the sky coordinates of the stars based on the loaded pixel coordinates and time.
        This method calculates the mean x and y pixel coordinates, converts them to RA/Dec using the
        raDec2Pix module, and stores the resulting SkyCoord object in self.stars_skycoord.
        
        Key arguments:
            - x: Array of x pixel coordinates (int or array-like in pixels)
            - y: Array of y pixel coordinates (int or array-like in pixels)
            - time: observation time in seconds (float)
            - module: Module number (int)
            - output: Output number (int)
        Returns:
            - stars_skycoord: SkyCoord object containing the RA/Dec coordinates of the stars.
            - time_obj: Astropy Time object representing the observation time in BJD (Barycentric Julian Date)
        """
        # Convert time from seconds to Kepler BJD (Barycentric Julian Date)
        time_bjd = time/3600/24 + 2454833  # in days (BJD)
        time_obj = Time(time_bjd, format="jd", scale="tdb")
        # Convert x and y from pixels to RA/Dec at the specified time
        ra, dec = PixelGWInjector.get_Pix2RaDec(
            module=np.ones_like(x)*module,
            output=np.ones_like(x)*output,
            row=x,
            column=y,
            time_obj=time_obj
        )
        # Create a SkyCoord object for the stars' coordinates
        stars_skycoord = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            frame='icrs'
        )
        return stars_skycoord, time_obj


    @staticmethod
    def calculate_deflections(gw_source, telcoord, stars_skycoord, time, time_obj, _m, _o):
        """
        Calculate the gravitational wave-induced deflections of stars' positions in the telescope frame.
        This method transforms the stars' sky coordinates to the telescope frame, applies the gravitational wave deflections,
        and then transforms the deflected coordinates back to sky coordinates and pixel coordinates.

        Key arguments:
            - gw_source: GWSource object representing the gravitational wave source.
            - telcoord: Telescope coordinates in the galactic frame (SkyCoord object).
            - stars_skycoord: SkyCoord object containing the stars' sky coordinates.
            - time: Array of observation times in seconds for each star (2D array).
            - time_obj: Astropy Time object representing the observation time in BJD (Barycentric Julian Date).
            - _m: Module number (int or array-like).
            - _o: Output number (int or array-like).
        Returns:
            - deflected_x: Array of deflected x pixel coordinates for each star (2D array).
            - deflected_y: Array of deflected y pixel coordinates for each star (2D array).
        """
        # Project the stars' sky coordinates to the telescope frame
        star_telecoord = estoiles.gw_calc.GWcalc.coordtransform(telcoord.galactic, stars_skycoord.galactic)
        # Calculate the deflections for each star time array
        # deflected_stars_telecoord = np.zeros((len(stars_skycoord), len(time[0]), 3))
        deflected_x = np.zeros_like(time)
        deflected_y = np.zeros_like(time)
        # For each star and its corresponding time array
        for star_i, time_array in enumerate(time):
            if gw_source is None:
                # If gw_source is None, no deflection is applied, return the original star coordinates
                deflected_star_telecoord = star_telecoord[:, star_i]
            else:
                # Calculate the deflections and add them to the original star coordinates
                deflected_star_telecoord = gw_source.dn(star_telecoord[:,star_i], time_array * u.s) + star_telecoord[:,star_i]
            # Convert the deflected star coordinates back to sky coordinates
            deflected_star_skycoord = estoiles.gw_calc.GWcalc.invcoordtransform(telcoord.galactic, deflected_star_telecoord.T)
            deflected_star_skycoord = deflected_star_skycoord.icrs
            # Convert the deflected star sky coordinates back to pixel coordinates
            m, o, r, c = PixelGWInjector.get_raDec2Pix(
                ra=deflected_star_skycoord.ra.deg,
                dec=deflected_star_skycoord.dec.deg,
                time_obj=time_obj
            )
            # Make sure the module and output are the same as the original values
            if not np.all(m == _m) or not np.all(o == _o):
                raise ValueError("Module and output numbers do not match the original values.")
            # Store the deflected pixel coordinates
            deflected_x[star_i] = r
            deflected_y[star_i] = c

        # return deflected_stars_telecoord
        return deflected_x, deflected_y


    @staticmethod
    def _deflect_star_single(args):
        """
        Helper function to calculate the deflection for a single star at a given time.
        This function is designed to be used with parallel processing.
        Key arguments:
            - args: A tuple containing the star index, time array, telescope coordinates, star coordinates in telescope frame,
                    gravitational wave source, time object, and module/output numbers.
        Returns:
            - star_i: Index of the star in the original list.
            - r: Row pixel coordinate of the deflected star.
            - c: Column pixel coordinate of the deflected star.
        """
        star_i, time_array, telcoord, star_telecoord_i, gw_source, time_obj, _m, _o = args
        if gw_source is None:
            # If gw_source is None, no deflection is applied, return the original star coordinates
            deflected_star_telecoord = star_telecoord_i
        else:
            # Calculate the deflection for the star at the given time
            deflected_star_telecoord = gw_source.dn(star_telecoord_i, time_array * u.s) + star_telecoord_i
        deflected_star_skycoord = estoiles.gw_calc.GWcalc.invcoordtransform(telcoord.galactic, deflected_star_telecoord.T).icrs
        m, o, r, c = PixelGWInjector.get_raDec2Pix(deflected_star_skycoord.ra.deg, deflected_star_skycoord.dec.deg, time_obj)
        if not np.all(m == _m) or not np.all(o == _o):
            raise ValueError("Module/output mismatch.")
        return star_i, r, c

    @staticmethod
    def calculate_deflections_parallel(gw_source, telcoord, stars_skycoord, time, time_obj, _m, _o):
        """
        Calculate the gravitational wave-induced deflections of stars' positions in the telescope frame using parallel processing.
        This method transforms the stars' sky coordinates to the telescope frame, applies the gravitational wave deflections,
        and then transforms the deflected coordinates back to sky coordinates and pixel coordinates.

        Key arguments:
            - gw_source: GWSource object representing the gravitational wave source.
            - telcoord: Telescope coordinates in the galactic frame (SkyCoord object).
            - stars_skycoord: SkyCoord object containing the stars' sky coordinates.
            - time: Array of observation times in seconds for each star (N_stars, N_times).
            - time_obj: Astropy Time object representing the observation time in BJD (Barycentric Julian Date).
            - _m: Module number (int or array-like).
            - _o: Output number (int or array-like).
        Returns:
            - deflected_x: Array of deflected x pixel coordinates for each star (N_stars, N_times).
            - deflected_y: Array of deflected y pixel coordinates for each star (N_stars, N_times).
        """
        # Project the stars' sky coordinates to the telescope frame
        star_telecoord = estoiles.gw_calc.GWcalc.coordtransform(telcoord.galactic, stars_skycoord.galactic)
        deflected_x = np.zeros_like(time)
        deflected_y = np.zeros_like(time)

        args = [(i, time[i], telcoord, star_telecoord[:, i], gw_source, time_obj, _m, _o) for i in range(len(time))]

        with ProcessPoolExecutor() as executor:
            for result in executor.map(PixelGWInjector._deflect_star_single, args):
                i, r, c = result
                deflected_x[i] = r
                deflected_y[i] = c

        return deflected_x, deflected_y



    def save_to_h5file(self, kicids, deflected_x, deflected_y, time, quarter, module, output, gw_params):
        """
        Save the deflected centroids for only the subset of stars corresponding to the specified module and output
        in the given quarter to a new HDF5 file. Also saves GW parameters under /auxiliary/gw_params/<gw_id> as pickled binary.
        
        Key arguments:
            - kicids: Array of KIC IDs for the stars.
            - deflected_x: Array of deflected x pixel coordinates for each star (N_stars, N_times).
            - deflected_y: Array of deflected y pixel coordinates for each star (N_stars, N_times).
            - time: Array of observation times in seconds for each star (N_stars, N_times).
            - quarter: Quarter number for the data.
            - gw_params: Dictionary containing the gravitational wave parameters.
        """
        input_path = Path(self.h5file_path)
        cat_id = input_path.stem  # 'test_cat' from 'test_cat.h5'
        gw_id = "gw_test_id"      # TODO: Replace with automatic or hash-based ID

        output_dir = input_path.parent / "gw_injected"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{cat_id}_{gw_id}.h5"

        # Copy input file to output if it doesn't exist yet
        if not output_path.exists():
            shutil.copy(input_path, output_path)

        with tables.open_file(output_path, mode='a') as f:
            group = getattr(f.root, f"{quarter}")
            pos_table = group.pos

            # modules = pos_table.cols.module[:]
            # outputs = pos_table.cols.output[:]

            # row_idx = 0
            # for i in range(len(modules)):
            #     if np.all(modules[i] == module) and np.all(outputs[i] == output):
            #         update_dict = {
            #             'x': np.expand_dims(deflected_x[row_idx] + 1., axis=0),
            #             'y': np.expand_dims(deflected_y[row_idx] + 1., axis=0),
            #         }
            #         pos_table.modify_rows(i, i+1, update_dict)
            #         # pos_table.modify_rows(i, i+1, {'x': deflected_x[row_idx], 'y': deflected_y[row_idx]})
            #         row_idx += 1
            x_all = pos_table.cols.x[:]
            y_all = pos_table.cols.y[:]
            modules = pos_table.cols.module[:]
            outputs = pos_table.cols.output[:]

            # Find rows matching the desired module/output
            matching_indices = [
                i for i in range(len(modules))
                if np.all(modules[i] == module) and np.all(outputs[i] == output)
            ]

            if len(matching_indices) != len(deflected_x):
                raise ValueError("Mismatch between deflected data and table rows")

            # Replace only matching rows
            for idx, i in enumerate(matching_indices):
                x_all[i] = deflected_x[idx]
                y_all[i] = deflected_y[idx]

            # Write full arrays back
            pos_table.cols.x[:] = x_all
            pos_table.cols.y[:] = y_all

            pos_table.flush()

            # Save GW params to auxiliary group
            if '/auxiliary' not in f:
                f.create_group('/', 'auxiliary')
            aux_group = f.root.auxiliary

            # Save the gw_params dict as a pickled blob
            param_bytes = pickle.dumps(gw_params)
            atom = tables.UInt8Atom()
            array = np.frombuffer(param_bytes, dtype='uint8')
            if hasattr(aux_group, gw_id):
                f.remove_node(aux_group, gw_id)
            f.create_array(aux_group, gw_id, array)

if __name__ == "__main__":
    h5file_path = Path("results/systematics/tests/test_cat/test_cat.h5")
    injector = PixelGWInjector(h5file_path, parallel=True)
