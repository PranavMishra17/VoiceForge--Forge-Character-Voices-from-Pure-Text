import sys
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InstallationValidator:
    """Wrapper class for CosyVoice installation validation"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {}
        
    def check_prerequisites(self):
        """Check if all required directories and files exist"""
        logger.info("Checking prerequisites...")
        
        required_paths = [
            'cosyvoice',
            'cosyvoice/third_party/Matcha-TTS',
            'pretrained_models/CosyVoice2-0.5B'
        ]
        
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
                logger.error(f"Missing required path: {path}")
        
        if missing_paths:
            logger.error("Prerequisites check failed!")
            return False
        
        logger.info("✓ All prerequisites found")
        return True
    
    def run_core_tests(self):
        """Execute core validation tests"""
        logger.info("\nRunning core validation tests...")
        
        try:
            from test_cosyvoice_core import run_all_tests
            return run_all_tests()
        except ImportError as e:
            logger.error(f"Failed to import core test module: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Core tests failed with error: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate validation report"""
        duration = (self.end_time - self.start_time).total_seconds()
        
        report = [
            "\n" + "="*60,
            "INSTALLATION VALIDATION REPORT",
            "="*60,
            f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {duration:.2f} seconds",
            "",
            "Results:",
        ]
        
        for test_name, result in self.results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            report.append(f"  {test_name}: {status}")
        
        all_passed = all(self.results.values())
        report.extend([
            "",
            "="*60,
            f"Overall Status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}",
            "="*60
        ])
        
        report_text = "\n".join(report)
        logger.info(report_text)
        
        # Save report to file
        try:
            os.makedirs('test_outputs', exist_ok=True)
            report_path = f"test_outputs/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)
            logger.info(f"\nReport saved to: {report_path}")
        except Exception as e:
            logger.warning(f"Could not save report file: {str(e)}")
        
        return all_passed
    
    def validate(self):
        """Main validation workflow"""
        self.start_time = datetime.now()
        
        logger.info("="*60)
        logger.info("VoiceForge Installation Validator")
        logger.info("="*60)
        
        # Step 1: Prerequisites
        prereq_result = self.check_prerequisites()
        self.results['Prerequisites Check'] = prereq_result
        
        if not prereq_result:
            logger.error("Cannot proceed - prerequisites not met")
            self.end_time = datetime.now()
            return self.generate_report()
        
        # Step 2: Core tests
        core_test_result = self.run_core_tests()
        self.results['Core Validation Tests'] = core_test_result
        
        # Generate final report
        self.end_time = datetime.now()
        return self.generate_report()


def main():
    """Entry point for validation"""
    validator = InstallationValidator()
    
    try:
        success = validator.validate()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error during validation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
