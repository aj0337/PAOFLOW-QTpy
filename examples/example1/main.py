from PAOFLOW_QTpy.conductor import Conductor


if __name__ == "__main__":
    simulation = Conductor()
    simulation.work_dir = "./al5.save"
    simulation.run()
