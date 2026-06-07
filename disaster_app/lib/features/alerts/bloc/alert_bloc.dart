import 'package:flutter_bloc/flutter_bloc.dart';
import '../../../core/services/alert_service.dart';
import '../../../core/models/alert_model.dart';

abstract class AlertEvent {}

class LoadAlerts extends AlertEvent {}

abstract class AlertState {}

class AlertLoading extends AlertState {}

class AlertLoaded extends AlertState {
  final List<Alert> alerts;
  AlertLoaded(this.alerts);
}

class AlertBloc extends Bloc<AlertEvent, AlertState> {
  final AlertService service;

  AlertBloc(this.service) : super(AlertLoading()) {
    on<LoadAlerts>((event, emit) async {
      emit(AlertLoading());
      final alerts = await service.getAlerts();
      emit(AlertLoaded(alerts));
    });
  }
}
