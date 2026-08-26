import os
import statistics

output_file = 'chr17_5kb_dSCC_conversion_factors_arrowhead.txt'
output_dir = 'output/chr17_5kb_arrowhead'
domain_folders = [os.path.join(output_dir, f"domain{i+1}") for i in range(len(os.listdir(output_dir)))]

dSCC_scores = []
conversion_factors = []

with open(output_file, 'w') as out_file:
    out_file.write("Domain_ID\tOptimal_dSCC\tOptimal_Conversion_Factor\n")

    for domain_folder in os.listdir(output_dir):
        domain_path = os.path.join(output_dir, domain_folder)

        if os.path.isdir(domain_path):
            log_file = os.path.join(domain_path, f'{domain_folder}_trained_log.txt')

            if os.path.exists(log_file):
                dSCC_score = None
                conversion_factor = None

                with open(log_file, 'r') as f:
                    for line in f:
                        if "Optimal dSCC" in line:
                            dSCC_score = line.split(":")[1].strip()
                        elif "Optimal conversion factor" in line:
                            conversion_factor = line.split(":")[1].strip()

                        if dSCC_score and conversion_factor:
                            break

                if dSCC_score and conversion_factor:
                    out_file.write(f"{domain_folder}\t{dSCC_score}\t{conversion_factor}\n")
                    dSCC_scores.append(float(dSCC_score))
                    conversion_factors.append(float(conversion_factor))
            else:
                print(f"Log file not found for {domain_folder}")

if conversion_factors:
    median_conversion_factor = statistics.median(conversion_factors)
    print(f"Median of Conversion Factors: {median_conversion_factor}")
else:
    print("No conversion factors found.")
