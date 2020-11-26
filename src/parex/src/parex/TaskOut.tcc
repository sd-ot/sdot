#include "TaskOut.h"

template<class T>
TaskOut<T>::TaskOut( TaskOut &&task_out ) : task( std::move( task_out.task ) ), data( task_out.data ) {
}

template<class T>
TaskOut<T>::TaskOut( Rc<Task> &&task, T *data ) : task( std::move( task ) ), data( data ) {
}

template<class T>
TaskOut<T>::TaskOut( Rc<Task> &&task ) : task( std::move( task ) ), data( reinterpret_cast<T *>( this->task->output_data() ) ) {
}

template<class T>
TaskOut<T>::TaskOut( T *data ) : data( data ) {
}

template<class T>
T *TaskOut<T>::operator->() const {
    return reinterpret_cast<T *>( data );
}

template<class T>
T &TaskOut<T>::operator*() const {
    return *operator->();
}
