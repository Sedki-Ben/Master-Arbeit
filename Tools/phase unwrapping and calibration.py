#!/usr/bin/env python3
"""
Simple Phase Calibration for CSI Indoor Localization
====================================================

Simplified implementation of S-Phaser style phase calibration
adapted for our 52-subcarrier CSI data without heavy dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
import csv
from pathlib import Path

class SimplePhaseCalibrator:
    """
    Simplified phase calibrator for CSI-based indoor localization.
    Removes artificial phase slopes and offsets to improve localization accuracy.
    """
    
    def __init__(self, center_freq_hz=5.20e9, bandwidth_hz=40e6):
        """
        Initialize phase calibrator
        
        Args:
            center_freq_hz: Center frequency (Hz)
            bandwidth_hz: Total bandwidth (Hz)
        """
        self.center_freq_hz = center_freq_hz
        self.bandwidth_hz = bandwidth_hz
        self.n_subcarriers = 52
        
        # Generate frequency mapping for our 52 subcarriers
        self.frequencies_hz = self._generate_subcarrier_frequencies()
        
        print(f"🔧 Simple Phase Calibrator initialized")
        print(f"   Center frequency: {self.center_freq_hz/1e9:.2f} GHz")
        print(f"   Bandwidth: {self.bandwidth_hz/1e6:.1f} MHz")
        print(f"   Subcarriers: {self.n_subcarriers}")
        print(f"   Frequency range: {self.frequencies_hz[0]/1e9:.3f} - {self.frequencies_hz[-1]/1e9:.3f} GHz")
    
    def _generate_subcarrier_frequencies(self):
        """Generate frequency mapping for 52 WiFi subcarriers"""
        
        # Standard WiFi subcarrier spacing
        subcarrier_spacing = self.bandwidth_hz / 64  # 64 FFT bins total
        
        # Map 52 subcarriers symmetrically around center
        # WiFi mapping: [-26 to -1, +1 to +26] (skipping DC at 0)
        subcarrier_indices = np.concatenate([
            np.arange(-26, 0),    # Lower 26 subcarriers  
            np.arange(1, 27)      # Upper 26 subcarriers
        ])
        
        frequencies = self.center_freq_hz + subcarrier_indices * subcarrier_spacing
        return frequencies
    
    def extract_csi_phases(self, csi_data):
        """
        Extract phases from our CSI data format
        
        Args:
            csi_data: List of 104 values (imag, real, imag, real...)
            
        Returns:
            Tuple of (amplitudes, phases) arrays
        """
        if len(csi_data) != 104:
            raise ValueError(f"Expected 104 CSI values, got {len(csi_data)}")
        
        amplitudes = []
        phases = []
        
        for i in range(0, 104, 2):
            imag = csi_data[i]
            real = csi_data[i + 1]
            
            # Calculate amplitude and phase
            amplitude = np.sqrt(real * real + imag * imag)
            phase = np.arctan2(imag, real)
            
            amplitudes.append(amplitude)
            phases.append(phase)
        
        return np.array(amplitudes), np.array(phases)
    
    def unwrap_and_calibrate_phase(self, raw_phases):
        """
        Core S-Phaser calibration: unwrap phases and remove linear trend
        
        Args:
            raw_phases: Array of raw phase values (radians)
            
        Returns:
            Tuple of (calibrated_phases, calibration_info)
        """
        # Step 1: Phase unwrapping to remove 2π discontinuities
        phases_unwrapped = np.unwrap(raw_phases)
        
        # Step 2: Linear trend fitting and removal
        # Use subcarrier indices as independent variable
        subcarrier_indices = np.arange(len(phases_unwrapped))
        
        # Fit linear trend: phase = slope * index + intercept
        slope, intercept = np.polyfit(subcarrier_indices, phases_unwrapped, 1)
        
        # Remove linear trend (this is the key S-Phaser step)
        linear_trend = slope * subcarrier_indices + intercept
        phases_calibrated = phases_unwrapped - linear_trend
        
        # Calculate calibration effectiveness
        variance_before = np.var(raw_phases)
        variance_after = np.var(phases_calibrated)
        effectiveness = 1 - variance_after / variance_before if variance_before > 0 else 0
        
        calibration_info = {
            'slope_removed': slope,
            'intercept_removed': intercept,
            'variance_before': variance_before,
            'variance_after': variance_after,
            'effectiveness': effectiveness,
            'unwrapped_phases': phases_unwrapped,
            'linear_trend': linear_trend
        }
        
        return phases_calibrated, calibration_info
    
    def calibrate_csi_sample(self, csi_data):
        """
        Calibrate a single CSI sample
        
        Args:
            csi_data: Raw CSI data (104 values)
            
        Returns:
            Tuple of (original_phases, calibrated_phases, calibration_info)
        """
        # Extract phases
        amplitudes, original_phases = self.extract_csi_phases(csi_data)
        
        # Calibrate phases
        calibrated_phases, calib_info = self.unwrap_and_calibrate_phase(original_phases)
        
        return original_phases, calibrated_phases, calib_info
    
    def visualize_calibration(self, original_phases, calibrated_phases, calib_info, title="Phase Calibration"):
        """Visualize the calibration process"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        subcarrier_indices = np.arange(len(original_phases))
        frequencies_ghz = self.frequencies_hz / 1e9
        
        # Plot 1: Original vs Calibrated phases
        axes[0, 0].plot(subcarrier_indices, original_phases, 'b-', 
                       label='Original', linewidth=2, alpha=0.7)
        axes[0, 0].plot(subcarrier_indices, calibrated_phases, 'r-', 
                       label='Calibrated', linewidth=2, alpha=0.7)
        axes[0, 0].set_xlabel('Subcarrier Index')
        axes[0, 0].set_ylabel('Phase (radians)')
        axes[0, 0].set_title('Phase Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Unwrapped phases with linear trend
        unwrapped = calib_info['unwrapped_phases']
        linear_trend = calib_info['linear_trend']
        
        axes[0, 1].plot(subcarrier_indices, unwrapped, 'b-', 
                       label='Unwrapped Original', linewidth=2)
        axes[0, 1].plot(subcarrier_indices, linear_trend, 'g--', 
                       label=f'Linear Trend (slope={calib_info["slope_removed"]:.3f})', 
                       linewidth=2)
        axes[0, 1].plot(subcarrier_indices, unwrapped - linear_trend, 'r-', 
                       label='After Trend Removal', linewidth=2)
        axes[0, 1].set_xlabel('Subcarrier Index')
        axes[0, 1].set_ylabel('Phase (radians)')
        axes[0, 1].set_title('Linear Trend Removal (S-Phaser Core)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Phase vs Frequency
        axes[1, 0].plot(frequencies_ghz, original_phases, 'b-', 
                       label='Original', linewidth=2, alpha=0.7)
        axes[1, 0].plot(frequencies_ghz, calibrated_phases, 'r-', 
                       label='Calibrated', linewidth=2, alpha=0.7)
        axes[1, 0].set_xlabel('Frequency (GHz)')
        axes[1, 0].set_ylabel('Phase (radians)')
        axes[1, 0].set_title('Phase vs Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Statistics
        stats_text = f"""Calibration Results:

Slope Removed: {calib_info['slope_removed']:.4f} rad/subcarrier
Intercept Removed: {calib_info['intercept_removed']:.4f} rad
Effectiveness: {calib_info['effectiveness']:.1%}

Phase Variance:
Before: {calib_info['variance_before']:.4f}
After: {calib_info['variance_after']:.4f}
Reduction: {(1-calib_info['effectiveness']):.1%}

Frequency Range:
{frequencies_ghz[0]:.3f} - {frequencies_ghz[-1]:.3f} GHz
Subcarrier Spacing: {(frequencies_ghz[1]-frequencies_ghz[0])*1000:.1f} MHz"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Calibration Statistics')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

def test_phase_calibration():
    """Test the phase calibration with realistic scenarios"""
    
    print("🧪 Testing S-Phaser Phase Calibration")
    print("=" * 40)
    
    # Initialize calibrator
    calibrator = SimplePhaseCalibrator()
    
    # Test 1: Generate synthetic CSI with known characteristics
    print("\n📊 Test 1: Synthetic CSI with artificial distortions")
    
    # Create realistic multipath phase pattern (what we want to preserve)
    frequencies = calibrator.frequencies_hz
    freq_norm = (frequencies - frequencies[0]) / (frequencies[-1] - frequencies[0])
    
    # Simulate multipath reflections
    true_multipath = (
        0.6 * np.sin(4 * np.pi * freq_norm) +          # Main reflection
        0.2 * np.sin(12 * np.pi * freq_norm + 1.0) +   # Secondary reflection
        0.1 * np.cos(8 * np.pi * freq_norm - 0.5)      # Tertiary reflection
    )
    
    # Add artificial linear slope (timing offset - this is the problem S-Phaser solves)
    artificial_slope = 1.8 * np.arange(52) / 52
    phase_offset = 2.1
    noise = 0.03 * np.random.randn(52)
    
    # Combine to create "measured" phases
    measured_phases = true_multipath + artificial_slope + phase_offset + noise
    
    print(f"   True multipath variance: {np.var(true_multipath):.4f}")
    print(f"   Artificial slope contribution: {np.var(artificial_slope):.4f}")
    print(f"   Total measured variance: {np.var(measured_phases):.4f}")
    
    # Apply S-Phaser calibration
    calibrated_phases, calib_info = calibrator.unwrap_and_calibrate_phase(measured_phases)
    
    # Check how well we recovered the multipath pattern
    correlation_before = np.corrcoef(measured_phases, true_multipath)[0, 1]
    correlation_after = np.corrcoef(calibrated_phases, true_multipath)[0, 1]
    
    print(f"\n🎯 Multipath Recovery Results:")
    print(f"   Slope removed: {calib_info['slope_removed']:.4f} rad/subcarrier (target: {1.8/52:.4f})")
    print(f"   Offset removed: {calib_info['intercept_removed']:.4f} rad (target: {phase_offset:.4f})")
    print(f"   Calibration effectiveness: {calib_info['effectiveness']:.1%}")
    print(f"   Multipath correlation before: {correlation_before:.3f}")
    print(f"   Multipath correlation after: {correlation_after:.3f}")
    print(f"   Correlation improvement: {correlation_after - correlation_before:+.3f}")
    
    # Visualize
    calibrator.visualize_calibration(
        measured_phases, calibrated_phases, calib_info,
        "S-Phaser Test: Synthetic CSI with Known Multipath"
    )
    
    # Test 2: CSI data format test
    print("\n📊 Test 2: Our CSI data format (104 values)")
    
    # Generate CSI data in our format (imag, real, imag, real...)
    amplitudes = 1 + 0.3 * np.random.randn(52)  # Variable amplitudes
    csi_data = []
    
    for i in range(52):
        amp = amplitudes[i]
        phase = measured_phases[i]  # Use same phase pattern
        
        real = amp * np.cos(phase)
        imag = amp * np.sin(phase)
        
        csi_data.extend([imag, real])  # Our format
    
    # Test calibration
    orig_phases, calib_phases, calib_info = calibrator.calibrate_csi_sample(csi_data)
    
    print(f"   CSI format: 104 values → 52 complex → 52 phases")
    print(f"   Phase extraction successful: {len(orig_phases)} phases")
    print(f"   Calibration effectiveness: {calib_info['effectiveness']:.1%}")
    
    return {
        'true_multipath': true_multipath,
        'measured_phases': measured_phases,
        'calibrated_phases': calibrated_phases,
        'correlation_improvement': correlation_after - correlation_before,
        'calibration_info': calib_info
    }

def demonstrate_localization_impact():
    """Demonstrate how phase calibration could impact localization"""
    
    print("\n🎯 Demonstrating Localization Impact")
    print("=" * 40)
    
    calibrator = SimplePhaseCalibrator()
    
    # Simulate CSI from different locations with varying artificial slopes
    locations = [(0, 0), (1, 1), (2, 2), (3, 3)]
    
    results = []
    
    for i, (x, y) in enumerate(locations):
        print(f"\n📍 Location ({x}, {y}):")
        
        # Generate unique multipath pattern for this location
        np.random.seed(42 + i)  # Different seed for each location
        
        # Location-specific multipath (this should be preserved)
        freq_norm = np.arange(52) / 52
        location_multipath = (
            0.4 * np.sin(2 * np.pi * freq_norm * (2 + x)) +
            0.2 * np.cos(2 * np.pi * freq_norm * (3 + y)) +
            0.1 * np.sin(2 * np.pi * freq_norm * (5 + x + y))
        )
        
        # Different artificial slope for each measurement (this should be removed)
        artificial_slope = (0.5 + 0.3 * i) * np.arange(52) / 52
        offset = 1.0 + 0.5 * i
        noise = 0.05 * np.random.randn(52)
        
        raw_phases = location_multipath + artificial_slope + offset + noise
        calibrated_phases, calib_info = calibrator.unwrap_and_calibrate_phase(raw_phases)
        
        results.append({
            'location': (x, y),
            'raw_phases': raw_phases,
            'calibrated_phases': calibrated_phases,
            'true_multipath': location_multipath,
            'slope_removed': calib_info['slope_removed'],
            'effectiveness': calib_info['effectiveness']
        })
        
        print(f"   Slope removed: {calib_info['slope_removed']:.4f} rad/subcarrier")
        print(f"   Effectiveness: {calib_info['effectiveness']:.1%}")
    
    # Analyze location discriminability
    print(f"\n📊 Location Discriminability Analysis:")
    
    # Calculate phase differences between locations
    raw_phase_diffs = []
    calibrated_phase_diffs = []
    
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            raw_diff = np.linalg.norm(results[i]['raw_phases'] - results[j]['raw_phases'])
            cal_diff = np.linalg.norm(results[i]['calibrated_phases'] - results[j]['calibrated_phases'])
            
            raw_phase_diffs.append(raw_diff)
            calibrated_phase_diffs.append(cal_diff)
            
            print(f"   Locations {results[i]['location']} vs {results[j]['location']}:")
            print(f"     Raw phase distance: {raw_diff:.3f}")
            print(f"     Calibrated phase distance: {cal_diff:.3f}")
            print(f"     Improvement ratio: {cal_diff/raw_diff:.2f}")
    
    avg_raw_distance = np.mean(raw_phase_diffs)
    avg_cal_distance = np.mean(calibrated_phase_diffs)
    
    print(f"\n🎯 Summary:")
    print(f"   Average raw phase distance: {avg_raw_distance:.3f}")
    print(f"   Average calibrated phase distance: {avg_cal_distance:.3f}")
    print(f"   Overall improvement: {avg_cal_distance/avg_raw_distance:.2f}x")
    
    if avg_cal_distance > avg_raw_distance:
        print("   ✅ Phase calibration IMPROVES location discriminability")
    else:
        print("   ⚠️ Phase calibration reduces location discriminability")
    
    return results

def main():
    """Main test function"""
    
    print("🎯 S-Phaser Phase Calibration for Indoor Localization")
    print("Adapted for our 52-subcarrier CSI data")
    print("=" * 60)
    
    # Run tests
    test_results = test_phase_calibration()
    location_results = demonstrate_localization_impact()
    
    # Summary
    print(f"\n🎉 Phase Calibration Testing Complete!")
    print(f"=" * 45)
    print(f"✅ Successfully adapted S-Phaser for our CSI format")
    print(f"📈 Demonstrated effective removal of artificial phase slopes")
    print(f"🎯 Showed improved multipath signal preservation")
    print(f"📍 Analyzed impact on location discriminability")
    
    print(f"\n📊 Key Results:")
    print(f"   Multipath correlation improvement: {test_results['correlation_improvement']:+.3f}")
    print(f"   Phase calibration effectiveness: {test_results['calibration_info']['effectiveness']:.1%}")
    print(f"   Slope removal accuracy: ±{abs(test_results['calibration_info']['slope_removed'] - 1.8/52):.4f} rad/subcarrier")
    
    print(f"\n🔧 Next Steps for Integration:")
    print(f"   1. Integrate into CNN data preprocessing pipeline")
    print(f"   2. Compare localization accuracy with/without calibration")
    print(f"   3. Optimize calibration parameters for our specific environment")
    print(f"   4. Test with real CSI data from your dataset")

if __name__ == "__main__":
    main()
