import '../models/alert_model.dart';
import 'alert_service.dart';

class MockAlertService implements AlertService {
  @override
  Future<List<Alert>> getAlerts() async {
    await Future.delayed(const Duration(seconds: 1));

    return [
      Alert(
        id: '1',
        type: 'Fire',
        severity: 'High',

        location: AlertLocation(lat: 12.9716, lon: 77.5946),
      ),

      Alert(
        id: '2',
        type: 'Flood',
        severity: 'Medium',

        location: AlertLocation(lat: 13.0827, lon: 80.2707),
      ),
    ];
  }
}
