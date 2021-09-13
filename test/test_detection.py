import os
import vipy
from vipy.util import remkdir
import heyvi


def test_face():
    im = vipy.image.Image(url='https://upload.wikimedia.org/wikipedia/en/d/d6/Friends_season_one_cast.jpg').mindim(256)
    d = heyvi.detection.FaceDetector()
    ims = d(im)
    assert len(ims.objects()) == 6
    print('[test_detection]: face detector passed')

    
def test_object():
    im = vipy.image.vehicles().mindim(256)
    d = heyvi.detection.ObjectDetector()
    ims = d(im)
    assert len(ims.objects()) == 273
    print('[test_detection]: object detector passed')
    

def _test_actor_association():
    raise ValueError('FIXME: legacy test case")

    from pycollector.admin.video import Video  # admin only
    from pycollector.admin.globals import backend

    videoid = ['16E4872D-94BA-4764-B47E-65DAFBA5A055',
               'A3A05DEF-4E8B-4650-B6D2-71BF43AD18D8',
               '2329CC6E-0C3C-4131-82CC-C0D97E014D28',
               '06F2C244-13B1-4432-BACE-50E7B7DB3031',
               '8B51B8F4-8563-4EE6-B07F-40FA5CBFC08F',
               '2A0E7D65-2516-4339-85C4-58E9723620E4',
               '4CEFF31A-EDB7-4965-AEAF-1011F0B7F6FD',
               '2153BF43-7400-404C-90F8-E4DC01FB1CD7',
               'AB1F0FEC-0792-4D63-B134-8FCBA770A4BA',
               'E8F54604-E424-4288-ABCE-3E250F02A606',
               'AFCACDBF-4D3D-47B7-8C26-9CC20A30D676',
               'F2AFB94C-24A1-4E8F-A9C7-9116724EE2B9',
               '2144C42F-F4FF-4026-A694-611A260A0A25',
               '155942E9-8737-4198-9D81-1CEC7B57FE65',
               '0A51105F-EBDB-4B6E-8E8E-339BDFCB0B90',
               'FF827732-7879-4A71-8AB5-C08955ACCA04',
               'F36203CA-60BC-40AD-9401-34A184C16D83',
               '5C778B7F-3A51-420A-8BF4-90340D2F8245',
               '40D34A1A-1452-46C4-8349-8DF6E839C488',
               'B5C773D4-A4FD-42DE-A835-D71C0A95BF82',
               '2462FAD0-978A-44FE-B76C-280CBDF28947',
               'BF50607F-A760-4A61-A59C-D7995D9CCC13',
               'A5CF88A6-50DD-4172-B584-2F449D3CB924',
               '962DD31D-16DE-41C6-B50B-D3A3C51E0376',
               '0BD99FE0-72B4-49C1-88BF-A6B1CA4962F9',
               '1D2B142B-C326-4220-88BE-CC8995AF36E9',
               'A44F1067-3407-4641-8082-FB97DAB690B0',
               '30DA050E-0A0E-436F-9B6C-4D75D6AD0733',
               'EBA0ECD8-FF0D-4243-85DF-817F9FB26CDD',
               '390B542F-DF37-40AD-A5D0-A62D64F3D686',
               'C173B159-63AD-4A72-A703-1759FF5D7BCA']
             
    V = [Video(id) for id in videoid]    

    V = vipy.util.scpload('scp://ma01-5200-0052:/tmp/57e1d8821c0df241.pkl')
    
    C = backend().collections()
    P = pycollector.detection.ActorAssociation()

    for v in V:
        for o in C.collection(v.project(), v.collection()).secondary_objects():
            target = o if o != 'Friend' else 'Person'
            vp = P(v.stabilize().savetmp(), target, dt=10)
            vp.crop(vp.trackbox().dilate(1.2)).annotate().saveas(os.path.join(remkdir('test_actor_association'), '%s.mp4' % v.videoid()))
            print(vp)

