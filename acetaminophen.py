# PBPK Model of acetaminophen
import numpy as np
import tomllib
from diffeqpy import ode

def transport(Q_tissue: float, V_tissue: float, P_tissue: float, C_tissue: float, C_source: float) -> float:
  """
  Standard PBPK Transport equation

  Args:
      Q_tissue (float): blood flow to tissue
      V_tissue (float): tissue volume
      P_tissue (float): tissue partition coefficient
      C_tissue (float): amount of substance in the tissue
      C_source (_type_): amount of tissue in the source compartment 

  Returns:
      float: rate of change as a result of the transport equation
  """
  return Q_tissue * (C_source - C_tissue/P_tissue) / V_tissue

def michaelismenten(C, v_max, k_m):
  return v_max * C / (k_m + C)

def conjugates(v_max_enz, phi, C, k_m_enz, k_i_enz, k_m_cf):
  return v_max_enz * phi * C / ((k_m_enz + C + C**2 / k_i_enz) * (k_m_cf + phi))

def oral_dose(D_oral, t, t_dose, TG, TP):
  return D_oral * (np.exp(-int(t > t_dose)*(t-t_dose) / TG) - np.exp(-int(t > t_dose)*(t-t_dose) / TP)) / (TG - TP)

def fraction(x):
  if x <= 1000:
    F_A = 0.0005 * x + 0.37
  else:
    F_A = 0.88
  return F_A

def blood_flow(p, BW):
  """
  Get and convert relative blood flow parameters 

  Args:
      p (iterable): parameter vector
      BW (float): body weight

  Returns:
      iterable: blood flow parameters
  """
  return p[2] * BW**(0.75) * p[3:9]

def volumes(p, BW):
  """
  Get volume parameters

  Args:
      p (iterable): parameter vector
      BW (float): body weight

  Returns:
      iterable: volume parameters
  """
  return p[9:17]*BW**(0.75)

def partitions(p):
  """
  Get partition coefficients

  Args:
      p (iterable): parameter vector

  Returns:
      iterable: partition coefficients
  """
  return p[17:35]


  
def acetaminophen_d(du, u, p, t):

  # Parameter decomposition
  D_oral = p[0]
  BW = p[1]
  Q_cardiac = p[2]*BW**(0.75) 
  Q_adipose, Q_muscle, Q_liver, Q_slow, Q_rapid, Q_renal = blood_flow(p,BW)
  V_arterial, V_adipose, V_muscle, V_liver, V_slow, V_rapid, V_renal, V_venous = volumes(p, BW)
  (P_adipose, P_muscle, P_liver, P_slow, P_rapid, P_renal, 
    P_adipose_G, P_muscle_G, P_liver_G, P_slow_G, P_rapid_G, P_renal_G,
    P_adipose_S, P_muscle_S, P_liver_S, P_slow_S, P_rapid_S, P_renal_S)  = partitions(p)

  # Absorption
  TG = p[35]
  TP = p[36]

  # Cytochrome p450
  k_m_cyp = p[37]
  v_max_cyp = p[38] * BW**(0.75)

  # Sulfation
  k_m_G = p[39]
  k_i_G = p[40]
  k_m_cfG = p[41]
  v_max_G = p[42] * BW**(0.75)

  # Glucuronidation
  k_m_S = p[43]
  k_i_S = p[44]
  k_m_cfS = p[45]
  v_max_S = p[46] * BW**(0.75)

  # Active hepatic transport
  k_m_G_mem = p[47]
  v_max_G_mem = p[48]
  k_m_S_mem = p[49]
  v_max_S_mem = p[50]

  # Cofactor synthesis
  k_syn_G = p[51]
  k_syn_S = p[52]

  # Renal clearance
  k_renal = p[53] * BW**(0.75)
  k_renal_G = p[54] * BW**(0.75)
  k_renal_S = p[55] * BW**(0.75)

  # Conversion mg to micromole
  mg_to_umol = p[56] # umol/mg

  D_iv = p[57]
  t_dose = p[58]

  # Metabolism by cytochrome p450 into NAPQI
  v_cyp = michaelismenten(u[3], v_max_cyp, k_m_cyp)

  # Metabolism into APAP-G using PAPS
  v_acetaminophen_G = conjugates(v_max_G, u[30],u[3],k_m_G,k_i_G,k_m_cfG)

  # Metabolism into APAP-S using UDPGA 
  v_acetaminophen_S = conjugates(v_max_S, u[31],u[3],k_m_S,k_i_S,k_m_cfS)


  # Acetaminophen
  ## Arterial Blood
  du[0] = transport(Q_cardiac, V_arterial, 1.0, u[0], u[8])
  ## Adipose
  du[1] = transport(Q_adipose, V_adipose, P_adipose, u[1], u[0])
  ## Muscle
  du[2] = transport(Q_muscle, V_muscle, P_muscle, u[2], u[0])
  ## Liver
  du[3] = transport(Q_liver, V_liver, P_liver, u[3], u[0]) - (v_cyp + v_acetaminophen_G + v_acetaminophen_S) / (V_liver) + (fraction(D_oral) * mg_to_umol / V_liver) * oral_dose(D_oral, t, t_dose, TG, TP)
  ## Slowly perfused
  du[4] = transport(Q_slow, V_slow, P_slow, u[4], u[0])
  ## Rapidly perfused
  du[5] = transport(Q_rapid, V_rapid, P_rapid, u[5], u[0])
  ## Kidneys
  du[6] = transport(Q_renal, V_renal, P_renal, u[6], u[0]) - k_renal * u[0] / V_renal 
  ## GI-tract
  du[7] = - oral_dose(D_oral, t, t_dose, TG, TP)
  ## Venous blood
  du[8] = (Q_adipose*u[1]/P_adipose + Q_muscle*u[2]/P_muscle + Q_liver*u[3]/P_liver + Q_slow*u[4]/P_slow + Q_rapid*u[5]/P_rapid + Q_renal*u[6]/P_renal - Q_cardiac*u[8])/V_venous + D_iv*mg_to_umol / V_venous
  ## Urine (for calculation)
  du[9] = k_renal * u[0]

  # Acetaminophen-G
  ## Arterial blood
  du[10] = transport(Q_cardiac, V_arterial, 1.0, u[10], u[18])
  ## Adipose
  du[11] = transport(Q_adipose, V_adipose, P_adipose_G, u[11], u[10])
  ## Muscle
  du[12] = transport(Q_muscle, V_muscle, P_muscle_G, u[12], u[10])
  ## Hepatocyte (conversion from acetaminophen)
  du[13] = v_acetaminophen_G - michaelismenten(u[13], v_max_G_mem, k_m_G_mem)
  ## Liver
  du[14] = transport(Q_liver, V_liver, P_liver_G, u[14], u[10]) + michaelismenten(u[13], v_max_G_mem, k_m_G_mem)/V_liver
  ## Slowly perfused
  du[15] = transport(Q_slow, V_slow, P_slow_G, u[15], u[10])
  ## Rapidly perfused
  du[16] = transport(Q_rapid, V_rapid, P_rapid_G, u[16], u[10])
  ## Kidneys
  du[17] = transport(Q_renal, V_renal, P_renal_G, u[17], u[10]) - k_renal_G * u[10] / V_renal
  ## Venous blood
  du[18] = (Q_adipose*u[11]/P_adipose_G + Q_muscle*u[12]/P_muscle_G + Q_liver*u[14]/P_liver_G + Q_slow*u[15]/P_slow_G + Q_rapid*u[16]/P_rapid_G + Q_renal*u[17]/P_renal_G - Q_cardiac*u[18])/V_venous
  ## Urine (for calculation)
  du[19] = k_renal_G *u[10]

  # Acetaminophen-S
  ## Arterial blood
  du[20] = transport(Q_cardiac, V_arterial, 1.0, u[20], u[28])
  ## Adipose
  du[21] = transport(Q_adipose, V_adipose, P_adipose_S, u[21], u[20])
  ## Muscle
  du[22] = transport(Q_muscle, V_muscle, P_muscle_S, u[22], u[20])
  ## Hepatocyte (conversion from acetaminophen)
  du[23] = v_acetaminophen_S - michaelismenten(u[23], v_max_S_mem, k_m_S_mem)
  ## Liver
  du[24] = transport(Q_liver, V_liver, P_liver_S, u[24], u[20]) + michaelismenten(u[23], v_max_S_mem, k_m_S_mem)/V_liver
  ## Slowly perfused
  du[25] = transport(Q_slow, V_slow, P_slow_S, u[25], u[20])
  ## Rapidly perfused
  du[26] = transport(Q_rapid, V_rapid, P_rapid_S, u[26], u[20])
  ## Kidneys
  du[27] = transport(Q_renal, V_renal, P_renal_S, u[27], u[20]) - k_renal_S* u[20] / V_renal
  ## Venous blood
  du[28] = (Q_adipose*u[21]/P_adipose_S + Q_muscle*u[22]/P_muscle_S + Q_liver*u[24]/P_liver_S + Q_slow*u[25]/P_slow_S + Q_rapid*u[26]/P_rapid_S + Q_renal*u[27]/P_renal_S - Q_cardiac*u[28])/V_venous
  ## Urine (for calculation)
  du[29] = k_renal_S *u[20]

  # Cofactors
  du[30] = -v_acetaminophen_G + k_syn_G * (1 - u[30])
  du[31] = -v_acetaminophen_S + k_syn_S * (1 - u[31])

  return du

COMPARTMENTS = {
  "arterial_blood": {
    "acetaminophen": 0, "acetaminophen_G": 10, "acetaminophen_S": 20
  },
  "adipose_tissue": {
    "acetaminophen": 1, "acetaminophen_G": 11, "acetaminophen_S": 21
  },
  "muscle": {
    "acetaminophen": 2, "acetaminophen_G": 12, "acetaminophen_S": 22
  },
  "liver": {
    "acetaminophen": 3, "acetaminophen_G": 14, "acetaminophen_S": 24
  },
  "hepatocyte": {
    "acetaminophen_G": 13, "acetaminophen_S": 23
  },
  "slowly_perfused": {
    "acetaminophen": 4, "acetaminophen_G": 15, "acetaminophen_S": 25
  },
  "rapidly_perfused": {
    "acetaminophen": 5, "acetaminophen_G": 16, "acetaminophen_S": 26
  },
  "kidneys": {
    "acetaminophen": 6, "acetaminophen_G": 17, "acetaminophen_S": 27
  },
  "GI": {
    "acetaminophen": 7
  },
  "venous_blood": {
    "acetaminophen": 8, "acetaminophen_G": 18, "acetaminophen_S": 28
  },
  "cofactor": {
    "UDPGA": 30, "PAPS": 31
  },
  "urine": {
    "acetaminophen": 9, "acetaminophen_G": 19, "acetaminophen_S": 29
  }
}

def _check_toml(data):

  # required keys:
  not_found = []
  for key in ["solution", "physiology", "volumes", "partition", "absorption", "metabolism", "transport", "cofactors", "clearance", "meta"]:
    if key not in data.keys():
      not_found.append(key)
  if not_found:
    raise ValueError(f"Could not find required keys: {not_found} in patient file")
  
  return True

def _check_physiology(physiology_data):

  not_found = []
  for key in ["body_weight", "cardiac_output"]:
    if key not in physiology_data.keys():
      not_found.append(key)
  if not_found:
    raise ValueError(f"Could not find required entries {not_found} in the physiology section of the file.")


def load_patient(patient_file):
  with open(patient_file, "rb") as f:
    patient_data = tomllib.load(f)
  
  # perform initial checks
  _check_toml(patient_data)

  # parse conditions
  u0 = np.array([0.0]*32)
  D_oral = 0.0
  conditions_compartments = patient_data["conditions"]
  for compartment, molecules in conditions_compartments.items():
    if compartment in COMPARTMENTS.keys():

      for molecule, value in molecules.items():
        if molecule in COMPARTMENTS[compartment].keys():
          u0[COMPARTMENTS[compartment][molecule]] = value

          if compartment == "GI" and molecule == "acetaminophen":
            # set initial dose
            D_oral = value
        
        else:
          print(f"W:\tNo molecule named {molecule} in compartment {compartment}.\nAvailable molecules:\n\t{COMPARTMENTS[compartment].keys()}")
    else:
      print(f"W\tNo compartment named {compartment} in model.\nAvailable compartments:\n\t{COMPARTMENTS.keys()}")
  
  # parse physiology
  physiology_data = patient_data["physiology"]
  _check_physiology(physiology_data)
  body_weight = physiology_data["body_weight"]
  cardiac_output = physiology_data["cardiac_output"]

  # parse volumes
  volumes = []
  volume_data = patient_data["volumes"]
  for compartment in ["arterial_blood", "adipose_tissue", "muscle", "liver", "slowly_perfused", "rapidly_perfused", "kidneys", "venous_blood"]:
    volumes.append(volume_data[compartment])
  
  # parse blood flows
  blood_flow = [cardiac_output]
  flow_data = patient_data["flow"]
  for compartment in ["adipose_tissue", "muscle", "liver", "slowly_perfused", "rapidly_perfused", "kidneys"]:
    if compartment not in flow_data.keys():
      raise ValueError(f"Blood flow for compartment {compartment} not found.")
    blood_flow.append(flow_data[compartment])

  # normalize blood flow fractions
  sum_fractional_flow = sum(blood_flow[1:])
  if sum_fractional_flow != 1:
    print("Fractional blood flow inputs do not sum to 1. Normalizing the fractional flows so they sum to 1")
    for i in range(1,len(blood_flow)):
      blood_flow[i] /= sum_fractional_flow
  
  # parse partitions
  partition_data = patient_data["partition"]
  partition_values = []
  for molecule in ["acetaminophen", "acetaminophen_G", "acetaminophen_S"]:
    if molecule not in partition_data.keys():
      raise ValueError(f"Partition data for [partition.{molecule}] not found")
    partition_molecule = partition_data[molecule]
    for compartment in ["adipose_tissue", "muscle", "liver", "slowly_perfused", "rapidly_perfused", "kidneys"]:
      if compartment not in partition_molecule.keys():
        raise ValueError(f"Partition entry for compartment {compartment} in [partition.{molecule}] not found")
      partition_values.append(partition_molecule[compartment])

  # parse parameters
  TG = patient_data["absorption"]["TG"]
  TP = patient_data["absorption"]["TP"]
  k_m_cyp = patient_data["metabolism"]["cyp"]["k_m"]
  v_max_cyp = patient_data["metabolism"]["cyp"]["v_max"]
  k_m_G = patient_data["metabolism"]["glucuronidation"]["k_m"]
  k_i_G = patient_data["metabolism"]["glucuronidation"]["k_i"]
  k_m_cfG = patient_data["metabolism"]["glucuronidation"]["k_m_cf"]
  v_max_G = patient_data["metabolism"]["glucuronidation"]["v_max"]

  k_m_S = patient_data["metabolism"]["sulfation"]["k_m"]
  k_i_S = patient_data["metabolism"]["sulfation"]["k_i"]
  k_m_cfS = patient_data["metabolism"]["sulfation"]["k_m_cf"]
  v_max_S = patient_data["metabolism"]["sulfation"]["v_max"]

  k_m_G_mem = patient_data["transport"]["acetaminophenG"]["k_m"]
  v_max_G_mem = patient_data["transport"]["acetaminophenG"]["v_max"]
  k_m_S_mem = patient_data["transport"]["acetaminophenS"]["k_m"]
  v_max_S_mem = patient_data["transport"]["acetaminophenS"]["v_max"]

  k_syn_G = patient_data["cofactors"]["k_syn_UDPGA"]
  k_syn_S = patient_data["cofactors"]["k_syn_PAPS"]

  k_renal = patient_data["clearance"]["k_renal"]
  k_renal_G = patient_data["clearance"]["k_renal_G"]
  k_renal_S = patient_data["clearance"]["k_renal_S"]

  parameters = [D_oral, body_weight] + blood_flow + volumes + partition_values + [
    TG, TP, k_m_cyp, v_max_cyp, k_m_G, k_i_G, k_m_cfG, v_max_G, k_m_S, k_i_S, k_m_cfS, v_max_S, 
    k_m_G_mem, v_max_G_mem, k_m_S_mem, v_max_S_mem, k_syn_G, k_syn_S, k_renal, k_renal_G, k_renal_S
  ] + [patient_data["meta"]["unit_conversions"]["acetaminophen"]["mg2umol"], 0.0, 0.0]

  tspan = tuple(patient_data["solution"]["timespan"])

  problem = ode.ODEProblem(acetaminophen_d, u0, tspan, np.array(parameters))

  return problem


def umol_to_mg(x, molecule):
  if molecule == "acetaminophen":
    return x / 6.6155
  elif molecule == "acetaminophen_G":
    return x / (1e3 / 327.29)
  elif molecule == "acetaminophen_S":
    return x / (1e3 / 231.23)


def additional_dose(initial_solution, initial_problem, dose, t_dose):
  params = initial_problem.p
  params[0] = initial_solution(t_dose)[COMPARTMENTS["GI"]["acetaminophen"]] + dose
  params[58] = t_dose

  u0 = np.array(initial_solution(t_dose))
  u0[COMPARTMENTS["GI"]["acetaminophen"]] += dose

  new_problem = ode.remake(initial_problem, u0=u0, p=params)
  return new_problem
