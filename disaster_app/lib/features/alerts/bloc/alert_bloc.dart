import 'package:flutter_bloc/flutter_bloc.dart';

import '../services/alert_service.dart';

import 'alert_event.dart';
import 'alert_state.dart';

class AlertBloc extends Bloc<AlertEvent, AlertState> {
  final AlertService service;

  AlertBloc(this.service) : super(AlertLoading()) {
    on<LoadAlerts>((event, emit) async {
      try {
        final alerts = await service.getAlerts();

        emit(AlertLoaded(alerts));
      } catch (e) {
        emit(AlertError("Failed to load alerts"));
      }
    });
  }
}
