import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('calibration',Path(__file__).resolve().parents[1]/'calibrate/fit.py')
calibration=importlib.util.module_from_spec(spec);spec.loader.exec_module(calibration)


def samples(typed=False):
    records=[dict(type='metadata',complete=False)]
    for i in range(6):
        row=dict(seat=0,observation=dict(time=i/30,phase='race',track_length_m=5795.843,
                 self=dict(x=i,y=0,speed_mps=20+i,compound='soft',pit_state=0,ers_soc_mj=3-i*.01),
                 weather=dict(condition='dry'),preview=dict(rows=[[15,.01]])),
                 action=dict(throttle=1.0,brake=0.0),fault=False)
        if typed:row['type']='decision'
        records.append(row)
    records.append(dict(type='export_complete',complete=True))
    return records


class CalibrationTests(unittest.TestCase):
    def fit_records(self,records):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'training.jsonl'
            p.write_text(''.join(json.dumps(r)+'\n' for r in records))
            return calibration.fit(p,'racing')

    def test_original_bytes_export_without_type(self):
        result=self.fit_records(samples())
        self.assertEqual(result['pairs'],5)
        self.assertGreater(result['parameters']['accel'],0)

    def test_typed_and_untyped_decisions_agree(self):
        self.assertEqual(self.fit_records(samples()),self.fit_records(samples(True)))

    def test_metadata_resets_previous_export(self):
        records=samples()[:-1]+samples()
        self.assertEqual(self.fit_records(records)['pairs'],10)

    def test_pit_exit_transition_is_not_driving_acceleration(self):
        records=samples();records[1]['observation']['self']['pit_state']=3
        self.assertEqual(self.fit_records(records)['pairs'],4)

    def test_fault_drops_pair_across_fault(self):
        records=samples();records[3]['fault']=True
        self.assertEqual(self.fit_records(records)['pairs'],3)


if __name__=='__main__':unittest.main()
