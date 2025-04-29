import torch
from .detector import generate_pmt_positions, pmt_collection_efficiency


class PhotonTransport:
    '''
    A class for simulating the transport of photons in a LArTPC detector.
    It calculates the number of photoelectrons detected by PMTs based on the number of photons emitted from each point in the detector.
    '''

    def __init__(self, config):
        '''
        Constructor for the OpticalPhysics class.

        Parameters
        ----------
        config : dict
            A collection of configuration parameters. It should contain:
            - 'light_yield': number of photons produced per MeV energy deposition.
        '''
        self.light_yield = config['PHYSICS']['light_yield']
        self.cathode_thickness = config['GEOMETRY']['TPC']['cathode_thickness']
        self.active_xrange = config['GEOMETRY']['TPC']['active_volume']['x']
        self.c = config['PHYSICS']['light_speed']
        self.c /= config['PHYSICS']['lar_refraction_index']
        self.time_resolution = config['SIMULATION']['TRUTH']['photon_time_resolution']
        self.ns2bin = 0.001 / self.time_resolution
        self.sigmoid_coeff = config['GEOMETRY']['PMT']['ce_angle_thres']
        self.device = 'cpu'
        self.debug_mode = config.get('DEBUG', False)

        lx = self.active_xrange[1] - self.active_xrange[0]
        ly = config['GEOMETRY']['TPC']['active_volume']['y']
        ly = ly[1] - ly[0]
        lz = config['GEOMETRY']['TPC']['active_volume']['z']
        lz = lz[1] - lz[0]
        spacing = config['GEOMETRY']['PMT']['sensor_spacing']
        self.gap_pmt_active = config['GEOMETRY']['PMT']['gap_pmt_active']
        self.pmt_positions, self.pmt_ids= generate_pmt_positions(lx=lx,
            ly=ly,lz=lz,
            spacing_y=spacing,spacing_z=spacing,
            gap_pmt_active=self.gap_pmt_active)
        self.sensor_radius = config['GEOMETRY']['PMT']['sensor_radius']

    def to(self,device):
        '''
        Move the model and all its torch.Tensor attributes to a specified device (CPU or GPU).

        Parameters
        ----------
        device : str
            The device to move the model to, e.g., 'cpu' or 'cuda'.
        '''
        self.device = device
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))


    def de_to_photons(self, de ,betagamma):
        '''
        Function to calculate the number of photons from deposited energy.

        Parameters
        ----------
        de : float32
            Deposited energy in MeV.
        betagamma : float32
            The relativistic factor (βγ) of the particle, where β = v/c and γ = 1/sqrt(1 - β²).

        Returns
        -------
        float32
            Number of photons produced from the deposited energy.
        '''
        return de * self.light_yield

    def get_pe(self, num_photons,points):
        '''
        Function to calculate the number of photoelectrons detected by PMTs.
        Parameters
        ----------
        num_photons : torch.Tensor
            A tensor of shape (N,) containing the number of photons emitted from each point.
        points : torch.Tensor
            A tensor of shape (N, 4) containing the (x, y, z, time) coordinates of each photon emission point.
        Returns
        -------
        torch.Tensor
            A tensor of shape (P,) containing the PMT IDs where photoelectrons were detected.
        torch.Tensor
            A tensor of shape (P,) containing the time bins of photoelectron detection.
        torch.Tensor
            A tensor of shape (P,) containing the number of photoelectrons detected at each PMT and time bin.
        '''
        
        pos_mask_v = [(points[:,0]>  self.cathode_thickness/2.) & (points[:,0]<(self.active_xrange[1]+self.gap_pmt_active)),
                    (points[:,0]<(-self.cathode_thickness/2.)) & (points[:,0]>(self.active_xrange[0]-self.gap_pmt_active))]

        pmt_mask_v = [self.pmt_positions[:,0]>self.cathode_thickness/2., self.pmt_positions[:,0]<(-self.cathode_thickness/2.)]

        pmt_data=[]
        res_id_v,res_t_v,res_n_v = [],[],[]
        for i in range(len(pos_mask_v)):
            if pos_mask_v[i].sum() < 1: continue
            pos = points[pos_mask_v[i]]
            if len(pos) < 1: continue
            pmt_pos = self.pmt_positions[pmt_mask_v[i]]
            pmt_ids = self.pmt_ids[pmt_mask_v[i]]

            nph_survived, tof = self.survived_photon2pmts(num_photons[pos_mask_v[i]], pos, pmt_pos)
            data=[]
            res_id = []
            res_t  = []
            res_n  = []
            for j, ppos in enumerate(pmt_pos):
                ts, ts_map = torch.unique(tof[:,j],return_inverse=True)

                res_t.append(ts)
                res_id.append(torch.ones(size=(len(ts),),dtype=torch.int32,device=self.device)*pmt_ids[j])

                n = torch.zeros(size=(len(ts),),dtype=torch.float32,device=self.device)
                n.index_add_(0, ts_map, nph_survived[:,j])
                res_n.append(n)

            res_id_v.append(torch.concat(res_id))
            res_t_v.append(torch.concat(res_t))
            res_n_v.append(torch.concat(res_n))

        return torch.concat(res_id_v), torch.concat(res_t_v), torch.concat(res_n_v)

    def survived_photon2pmts(self, nph, pos, pmt_pos):
        '''
            Function to calculate the number of photoelectrons detected by PMTs.
            Parameters
            ----------
            num_photons : torch.Tensor
                A tensor of shape (N,) containing the number of photons emitted from each point.
            points : torch.Tensor
                A tensor of shape (N, 4) containing the (x, y, z, time) coordinates of each photon emission point.
            Returns
            -------
            nph_survived: torch.Tensor
                A tensor of shape (N, M) containing the number of photons that survived the propagation to the M PMTs.
            tof: torch.Tensor
                A tensor of shape (N, M) containing the time of flight for each photon to each PMT.
        '''

        r, arccos, number_frac = self.propagate_photon2pmts(pos[:, :3], pmt_pos)
        tof = ((r.T / self.c + pos[:, 3]) * self.ns2bin + 0.5).T.to(torch.int32)
        # to verify tof need the float value, otherwise all 0
        #tof = (r.T / self.c + pos[:, 3]).T


        ce = pmt_collection_efficiency(arccos, sigmoid_coeff=self.sigmoid_coeff)
        return nph.unsqueeze(1) * number_frac * ce, tof

    def propagate_photon2pmts(self, photon_pos, pmt_pos):
        '''
        Function to calculate the relative spatial relations between the photon and pmt positions.
        Parameters
        ----------
        photon_pos : torch.Tensor
            A tensor of shape (N, 3) containing the photon emission points.
        pmt_pos : torch.Tensor
            A tensor of shape (M, 3) containing the pmt coordinates.
        Returns
        -------
        torch.Tensor
            A tensor of shape (N, M) containing the euclidean distances between the photon_pos and pmt_pos.
        torch.Tensor
            A tensor of shape (N, M) containing the angle of photons w.r.t PMT normal seen by each PMT
        torch.Tensor
            A tensor of shape (N, M) containing solid angle spanned by each PMT as seen from the photons.
        '''
        r = torch.cdist(photon_pos, pmt_pos)
        dx = torch.abs(photon_pos[:, None, 0] - pmt_pos[None, :, 0])
        #sin = torch.clamp(torch.abs(dx / r), max=1.0)
        #arcsin = torch.asin(sin)
        ### Bug-fix: pmts are placed in y-z plane, e.g. fixed x coordinates!
        cos = torch.clamp(torch.abs(dx / r), max=1.0)
        arccos = torch.acos(cos)
        #number_frac = torch.atan(self.sensor_radius * torch.sqrt(1 - sin**2) / r) / torch.pi
        factor = torch.sqrt(cos**2 * self.sensor_radius**2 + r**2)
        number_frac = 0.5 * ( 1. - r / factor )

        if self.debug_mode and torch.isnan(arccos).any():
            mask = torch.isnan(arccos)
            raise ValueError("arccos is NaN, r input: ", r[mask],
                             " dx input: ", dx[mask], " angle input: ",
                             arccos[mask], " number_frac: ", number_frac[mask])
            
        return r, arccos, number_frac
