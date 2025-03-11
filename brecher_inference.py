import sys
import matplotlib.pyplot as plt
from tools import ChatterDetector

def main(csv_path, rpm):
    # Initialize detector
    detector = ChatterDetector(rpm)
    
    # Load and process data
    detector.load_data(csv_path)
    ci_times, ci_values = detector.calculate_chatter_indicators()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Raw acceleration plot
    plt.subplot(2, 1, 1)
    plt.plot(detector.time, detector.accelX, label='X')
    plt.plot(detector.time, detector.accelY, label='Y')
    plt.title('Filtered Acceleration Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    
    # Chatter indicator plot
    plt.subplot(2, 1, 2)
    plt.plot(ci_times, ci_values, 'b-', label='Chatter Indicator')
    plt.axhline(detector.chatter_threshold, color='r', linestyle='--', label='Threshold')
    plt.title(f'Chatter Detection @ {rpm} RPM')
    plt.xlabel('Time (s)')
    plt.ylabel('CI Value')
    plt.ylim(0, 1.2)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python brecher_inference.py <csv_path> <rpm>")
    #     sys.exit(1)
    
    # csv_path = "C:/roland_anomalydetection/Legacy Code/Roland_BadCuttertests/EBI_Roland_Good_Cutter/EBI_Roland_Good_Cutter/70002.csv"
    csv_path = "C:/LJM-Data-Collection/csv/8500_2_timecalc.csv"
    # csv_path = "8500_trunc.csv"
    rpm = 8500
    main(csv_path, rpm)